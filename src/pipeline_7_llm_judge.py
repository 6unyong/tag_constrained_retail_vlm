"""
Task 19: LLM-as-a-Judge — Multimodal Pairwise Evaluation (v2)
==============================================================
Evaluates MOP captions against Baseline using a PAIRWISE comparison.
The judge receives the actual image + both captions and decides which
is more (A) visually accurate, (B) factually complete, (C) appropriate
in handling uncertainty.

Design rationale (v2 changes):
- PAIRWISE instead of absolute scoring → avoids scale bias
- MULTIMODAL: image sent alongside captions → judge sees what VLM saw
- NO circular Ground Truth: judge evaluates against the image, not L1-L4 tags
- Caption order randomised per item to prevent position bias

10K Resilience:
- Dict-based checkpoint: already-judged pairs skipped on re-run
- Per-image error handling + exponential backoff on 503
- Auto-save after every batch
- Semaphore-bounded async concurrency for throughput
"""
import asyncio
import base64
import os
import json
import random
import argparse
from google import genai
from google.genai import types
from dotenv import load_dotenv

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)

load_dotenv()

DEFAULT_BASELINE  = "data/cache/baseline_captions.json"
IMG_DIR           = "data/processed"
DEFAULT_OUT_PATH  = "data/eval_results/llm_judge_scores.json"
ERROR_LOG         = "data/cache/error_log.txt"
DEFAULT_MODEL     = "gemini-2.5-flash"

AUTOSAVE_INTERVAL = 5
CONCURRENCY       = 3


def log_error(img_name: str, error: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[pipeline_7] {img_name} | {type(error).__name__}: {error}\n")


def load_existing_cache(out_path: str = DEFAULT_OUT_PATH) -> dict:
    """Load previously judged results as {image_file: result} dict."""
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {r["image_file"]: r for r in existing.get("detailed_scores", [])}
    return {}


def save_results(cache: dict, out_path: str = DEFAULT_OUT_PATH):
    """Compute aggregate win-rates and persist results."""
    scores = list(cache.values())
    valid  = [s for s in scores if "preference" in s]

    mop_wins      = sum(1 for s in valid if s["preference"] == "MOP")
    baseline_wins = sum(1 for s in valid if s["preference"] == "Baseline")
    ties          = sum(1 for s in valid if s["preference"] == "Tie")
    n = max(len(valid), 1)

    # Per-dimension win rates
    dim_keys = ["visual_accuracy_winner", "factual_completeness_winner",
                "uncertainty_handling_winner"]
    dim_mop  = {k: sum(1 for s in valid if s.get(k) == "MOP") / n for k in dim_keys}

    os.makedirs("data/eval_results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_type": "pairwise_multimodal",
            "n_evaluated": len(valid),
            "overall_win_rate": {
                "MOP_wins":      mop_wins,
                "Baseline_wins": baseline_wins,
                "Ties":          ties,
                "MOP_win_pct":   round(mop_wins / n * 100, 1),
            },
            "dimension_win_rates": {
                k.replace("_winner", ""): round(v * 100, 1)
                for k, v in dim_mop.items()
            },
            "detailed_scores": scores,
        }, f, indent=4, ensure_ascii=False)


def _img_to_b64(img_path: str) -> str | None:
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def pairwise_judge(
    client, img_b64: str, caption_a: str, caption_b: str,
    label_a: str, label_b: str, model_name: str
) -> dict | None:
    """
    Ask Gemini to compare two captions given the actual image.
    caption_a / caption_b are presented as 'Caption 1' / 'Caption 2'
    to avoid label bias; caller tracks which is MOP vs Baseline.
    """
    prompt = f"""You are an expert evaluator for retail image captioning systems.

You will be shown a RETAIL IMAGE and TWO CAPTIONS written about it.
Your task is to decide which caption is BETTER on three dimensions:

1. VISUAL ACCURACY — Does the caption accurately describe what is VISIBLE in the image?
   (Penalise any invented brand names, products, or facts not supported by the image.)

2. FACTUAL COMPLETENESS — Does the caption mention the KEY elements visible in the image?
   (Scene type, main products/displays, seasonal themes if present, layout.)

3. UNCERTAINTY HANDLING — When something is NOT clearly visible (stock level, price, tidiness),
   does the caption appropriately decline to state it, rather than guessing?

CAPTION 1:
{caption_a}

CAPTION 2:
{caption_b}

Return ONLY valid JSON in this exact format:
{{
  "preference": "<Caption 1 | Caption 2 | Tie>",
  "visual_accuracy_winner":        "<Caption 1 | Caption 2 | Tie>",
  "factual_completeness_winner":   "<Caption 1 | Caption 2 | Tie>",
  "uncertainty_handling_winner":   "<Caption 1 | Caption 2 | Tie>",
  "reasoning": "<2-3 sentence explanation of the overall preference>"
}}"""

    img_part = types.Part.from_bytes(
        data=base64.b64decode(img_b64),
        mime_type="image/jpeg",
    )

    max_retries = 4
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=[img_part, prompt],
                config=types.GenerateContentConfig(temperature=0.1),
            )
            text = response.text or ""
            if "```json" in text:
                text = text.split("```json\n")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```\n")[1].split("```")[0]
            result = json.loads(text.strip())

            # Remap "Caption 1 / Caption 2" → actual labels (MOP / Baseline)
            def remap(val: str) -> str:
                if "1" in val:   return label_a
                if "2" in val:   return label_b
                return "Tie"

            return {
                "preference":                  remap(result.get("preference", "Tie")),
                "visual_accuracy_winner":      remap(result.get("visual_accuracy_winner", "Tie")),
                "factual_completeness_winner": remap(result.get("factual_completeness_winner", "Tie")),
                "uncertainty_handling_winner": remap(result.get("uncertainty_handling_winner", "Tie")),
                "reasoning":                   result.get("reasoning", ""),
            }

        except Exception as e:
            err_str = str(e)
            is_503  = "503" in err_str or "UNAVAILABLE" in err_str
            if is_503 and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [RETRY] 503 overload. Retrying in {wait}s ({attempt}/{max_retries})...")
                await asyncio.sleep(wait)
            else:
                print(f"  [ERROR] Gemini judge call failed: {e}")
                return None


async def _judge_single_image(
    img_name: str, mop_map: dict, base_map: dict,
    client, model_name: str, semaphore: asyncio.Semaphore,
) -> dict | None:
    """
    Process a single image judgment with semaphore-bounded concurrency.
    Returns the result dict (with image_file, captions, preference) or None.
    """
    mop_item  = mop_map[img_name]
    base_item = base_map[img_name]

    mop_caption  = mop_item.get("FINAL_CAPTION", "")
    base_caption = base_item.get("FINAL_CAPTION", "")

    img_path = os.path.join(IMG_DIR, img_name)
    img_b64  = _img_to_b64(img_path)

    if not img_b64:
        print(f"  [SKIP] Image file not found: {img_path}")
        return None

    print(f"Judging pair: {img_name}...")

    # Randomise caption order to prevent position bias
    if random.random() < 0.5:
        cap_a, cap_b = mop_caption, base_caption
        label_a, label_b = "MOP", "Baseline"
    else:
        cap_a, cap_b = base_caption, mop_caption
        label_a, label_b = "Baseline", "MOP"

    async with semaphore:
        result = await pairwise_judge(
            client, img_b64, cap_a, cap_b, label_a, label_b, model_name
        )

    if result:
        result["image_file"]    = img_name
        result["mop_caption"]   = mop_caption
        result["base_caption"]  = base_caption

        pref = result["preference"]
        print(f"  Preference: {pref} | "
              f"VisAcc: {result['visual_accuracy_winner']} | "
              f"FactComp: {result['factual_completeness_winner']} | "
              f"UncertH: {result['uncertainty_handling_winner']}")
        print(f"  Reasoning: {result['reasoning'][:100]}...")
        return result
    else:
        log_error(img_name, Exception("No result returned from judge"))
        return None


async def run_llm_judge():
    parser = argparse.ArgumentParser(description="Multimodal Pairwise LLM-as-a-Judge")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N pairs to judge (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for K-selection ablation (e.g. 'k2' or 'k4').")
    args = parser.parse_args()

    suffix_str = f"_{args.suffix}" if args.suffix else ""
    baseline_path = DEFAULT_BASELINE
    mop_path      = f"data/cache/final_captions{suffix_str}.json"
    out_path      = f"data/eval_results/llm_judge_scores{suffix_str}.json"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY in .env to run LLM-as-a-Judge.")
        return

    client     = genai.Client()
    model_name = args.model
    print(f"[LLM JUDGE v2] Multimodal Pairwise | Model: {model_name}")
    print(f"[PATHS] Baseline: {baseline_path} | Output: {out_path}")

    # Load both caption sets
    if not os.path.exists(mop_path):
        raise FileNotFoundError(f"MOP captions not found: {mop_path}")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline captions not found: {baseline_path}")

    with open(mop_path,       "r", encoding="utf-8") as f: mop_data      = json.load(f)
    with open(baseline_path,  "r", encoding="utf-8") as f: baseline_data = json.load(f)

    mop_map  = {i["image_file"]: i for i in mop_data      if "image_file" in i}
    base_map = {i["image_file"]: i for i in baseline_data if "image_file" in i}
    common   = sorted(set(mop_map) & set(base_map))
    print(f"[PAIRS] {len(common)} image pairs found (MOP + Baseline).")

    # Checkpoint
    cache        = load_existing_cache(out_path)
    already_done = len(cache)
    if already_done > 0:
        print(f"[CHECKPOINT] {already_done} pairs already judged. Resuming...")

    unjudged = [img for img in common if img not in cache]

    if args.sample and args.sample > already_done:
        remaining = args.sample - already_done
        if remaining < len(unjudged):
            unjudged = random.sample(unjudged, remaining)
            print(f"[SAMPLE] Sampled {len(unjudged)} pairs "
                  f"(target: {args.sample}, done: {already_done}).")

    print(f"\n--- Pairwise Multimodal Evaluation ({len(unjudged)} pairs to judge) ---")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    new_count = 0

    # Process in batches to allow periodic auto-save
    batch_size = CONCURRENCY * 5
    for batch_start in range(0, len(unjudged), batch_size):
        batch = unjudged[batch_start: batch_start + batch_size]

        tasks = [
            _judge_single_image(
                img_name, mop_map, base_map, client, model_name, semaphore
            )
            for img_name in batch
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None:
                cache[result["image_file"]] = result
                new_count += 1

        # Auto-save after each batch
        save_results(cache, out_path)
        print(f"  [AUTOSAVE] Batch complete. New this session: {new_count} | "
              f"Total: {already_done + new_count}/{len(common)}")

    # Final save + summary
    save_results(cache, out_path)
    valid = [s for s in cache.values() if "preference" in s]
    n = max(len(valid), 1)

    mop_wins  = sum(1 for s in valid if s["preference"] == "MOP")
    base_wins = sum(1 for s in valid if s["preference"] == "Baseline")
    ties      = sum(1 for s in valid if s["preference"] == "Tie")

    print("\n" + "=" * 50)
    print("  Pairwise LLM Judge -- Final Results")
    print("=" * 50)
    print(f"  MOP wins:      {mop_wins:>4} ({mop_wins/n*100:.1f}%)")
    print(f"  Baseline wins: {base_wins:>4} ({base_wins/n*100:.1f}%)")
    print(f"  Ties:          {ties:>4} ({ties/n*100:.1f}%)")
    print(f"  Total judged:  {len(valid)}")
    print("=" * 50)
    print(f"\nJudged this session: {new_count} | Total in cache: {len(cache)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_llm_judge())
