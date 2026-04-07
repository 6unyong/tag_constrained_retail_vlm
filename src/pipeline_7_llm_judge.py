"""
Task 19: LLM-as-a-Judge via Gemini Pro
Uses a powerful reasoning model to evaluate the generated captions for
Accuracy, Relevance, and Absence Handling.

10K Resilience features:
- Dict-based checkpoint: already-judged images are skipped on re-run
  (prevents double Gemini billing on partial runs)
- True async via client.aio (replaces blocking asyncio.to_thread)
- Per-image error handling
"""
import asyncio
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

OUT_PATH = "data/eval_results/llm_judge_scores.json"
ERROR_LOG = "data/cache/error_log.txt"


def log_error(img_name: str, error: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[pipeline_7] {img_name} | {type(error).__name__}: {error}\n")


def load_existing_cache() -> dict:
    """Load previously judged results as {image_file: result} dict."""
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {r["image_file"]: r for r in existing.get("detailed_scores", [])}
    return {}


def save_results(cache: dict):
    """Compute aggregate averages and persist results."""
    scores = list(cache.values())
    valid = [s for s in scores if "accuracy_score" in s]

    avg_acc = sum(s["accuracy_score"] for s in valid) / max(len(valid), 1)
    avg_rel = sum(s["relevance_score"] for s in valid) / max(len(valid), 1)
    avg_abs = sum(s["absence_handling_score"] for s in valid) / max(len(valid), 1)

    os.makedirs("data/eval_results", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "avg_accuracy": round(avg_acc, 2),
            "avg_relevance": round(avg_rel, 2),
            "avg_absence_handling": round(avg_abs, 2),
            "n_evaluated": len(valid),
            "detailed_scores": scores,
        }, f, indent=4)


async def judge_caption(client, caption: str, gt_data: dict, model_name: str) -> dict | None:
    prompt = f"""You are an expert academic evaluator, acting as "LLM-as-a-Judge" for an image captioning paper.
    
We generated a caption for a retail grocery image using a constrained deterministic pipeline.
Your job is to rate the caption from 1 to 10 on three metrics:

1. ACCURACY: Does the caption strictly only describe the facts listed in the Ground Truth without hallucinating external brands or objects?
2. RELEVANCE: Is the caption natural, coherent, and descriptively relevant for a store manager?
3. ABSENCE_HANDLING: If any Ground Truth facts say 'Ambiguous' or 'Unknown', did the caption explicitly decline to guess and mention its uncertainty?

Ground Truth Facts:
{json.dumps(gt_data, indent=2)}

Generated Caption:
{caption}

Return the results IN PURE JSON FORMAT:
{{
  "accuracy_score": <int 1-10>,
  "accuracy_reason": "<string explanation>",
  "relevance_score": <int 1-10>,
  "relevance_reason": "<string>",
  "absence_handling_score": <int 1-10>,
  "absence_reason": "<string>"
}}
"""
    try:
        # True async call — avoids blocking the event loop
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        text = response.text or ""
        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json\n")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```\n")[1].split("```")[0]

        return json.loads(text.strip())

    except Exception as e:
        print(f"  [ERROR] Gemini judge call failed: {e}")
        return None


async def run_llm_judge():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY in .env to run LLM-as-a-Judge.")
        return

    client = genai.Client()
    model_name = "gemini-3.1-pro-preview"

    with open("data/cache/final_captions.json", "r", encoding="utf-8") as f:
        captions_data = json.load(f)

    # --- Checkpoint: skip already-judged images (prevents double Gemini billing) ---
    cache = load_existing_cache()
    already_done = len(cache)
    if already_done > 0:
        print(f"[CHECKPOINT] {already_done} captions already judged. Resuming...")

    print(f"\n--- Running Gemini Pro (LLM-as-a-Judge) Evaluation ---")
    new_count = 0

    for item in captions_data:
        img_name = item.get("image_file")
        caption = item.get("FINAL_CAPTION")

        if img_name in cache:
            continue  # Already judged — skip Gemini call

        print(f"Judging {img_name}...")

        gt = {
            "L0_Context": item.get("global_context"),
            "L1_Scene": item.get("L1_scene"),
            "L2_Fixtures": item.get("L2_fixtures"),
            "L3_Products": item.get("L3_products"),
            "L4_Attributes": item.get("L4_attributes"),
        }

        score_json = await judge_caption(client, caption, gt, model_name)

        if score_json:
            score_json["image_file"] = img_name
            cache[img_name] = score_json
            new_count += 1

            print(
                f"  Accuracy: {score_json.get('accuracy_score')}/10 | "
                f"Relevance: {score_json.get('relevance_score')}/10 | "
                f"Absence: {score_json.get('absence_handling_score')}/10"
            )
            print(f"  Reason (Acc): {score_json.get('accuracy_reason','')[:80]}...")

            # Save after every judgement (cheap operation, prevents any loss)
            save_results(cache)
        else:
            log_error(img_name, Exception("No score returned from judge"))

    # Final save + summary
    save_results(cache)
    valid = [s for s in cache.values() if "accuracy_score" in s]
    if valid:
        print("\n==================================")
        print("[REPORT] LLM-as-a-Judge Final Averages:")
        print(f"  Accuracy:         {sum(s['accuracy_score'] for s in valid)/len(valid):.2f} / 10")
        print(f"  Relevance:        {sum(s['relevance_score'] for s in valid)/len(valid):.2f} / 10")
        print(f"  Absence Handling: {sum(s['absence_handling_score'] for s in valid)/len(valid):.2f} / 10")
        print(f"  Total evaluated:  {len(valid)}")
        print("==================================")

    print(f"\nJudged this session: {new_count} | Total in cache: {len(cache)}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run_llm_judge())
