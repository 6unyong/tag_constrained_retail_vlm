"""
Task 10: L3 Product Tagging
Step 1 — Send L1/L2 context + Kanops mappings to Gemini Flash -> get dynamic product keyword list
Step 2 — Use those keywords as CLIP zero-shot prompts -> produce L3 probabilistic tags

10K Resilience features:
- Dict-based checkpoint: already-tagged images are skipped on re-run (no double Gemini billing)
- Per-image try/except: one bad image won't kill the entire run
- Auto-save every 50 images to prevent data loss on crash/OOM
- Async semaphore: CONCURRENCY images processed in parallel for speed
- UTF-8 stdout forced: prevents UnicodeEncodeError on Korean Windows CMD
- Kanops fallback: if Gemini returns 0 keywords, use top GS1 categories as CLIP prompts
"""
import os
import sys
import json
import asyncio
import time
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# ── Force UTF-8 stdout to prevent UnicodeEncodeError on Windows Korean locale ──
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
if sys.stderr.encoding != "utf-8":
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

import torch
from paddleocr import PaddleOCR

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client()
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

# ── Constants ──
IN_PATH = "data/cache/l1_l2_tag_results.json"
OUT_PATH = "data/cache/l1_l2_l3_tag_results.json"
ERROR_LOG = "data/cache/error_log.txt"
AUTOSAVE_INTERVAL = 50

# Number of images to process concurrently (Gemini + OCR in parallel batches)
# Lowered to 3 (from 5) to reduce 503 overload errors from Gemini API.
CONCURRENCY = 3

# NOTE: No Kanops fallback — if Gemini returns 0 keywords, l3_source is set to "empty"
# so the captioning stage can skip L3 constraints and avoid hallucination.


def safe_print(msg: str):
    """Print with ASCII fallback to prevent UnicodeEncodeError on Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── Pydantic schema for Gemini response ──
class DynamicProductKeywords(BaseModel):
    keywords: List[str] = Field(
        ..., description="List of specific product names likely visible in this retail scene"
    )

def log_error(img_path: str, stage: str, error: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[pipeline_3b|{stage}] {img_path} | {type(error).__name__}: {error}\n")

def load_existing_cache() -> dict:
    """
    Load previously saved L3 results as {image_path: result} for checkpoint resumption.
    This is the primary guard against Gemini double-billing.
    """
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {item["image_path"]: item for item in existing}
    return {}

def save_cache(cache: dict):
    os.makedirs("data/cache", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(cache.values()), f, indent=4, ensure_ascii=False)

def extract_ocr_text(image_path: str) -> list:
    """Extract visible text from image using PaddleOCR. Safely handles None results."""
    try:
        result = ocr.ocr(image_path, cls=True)
    except Exception:
        return []
    texts = []
    # PaddleOCR can return None or [[]] on blank/corrupt images
    if not result or not result[0]:
        return texts
    try:
        for line in result[0]:
            if line is None or len(line) < 2:
                continue
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:
                texts.append({"text": text, "confidence": round(float(confidence), 3)})
    except (TypeError, AttributeError):
        pass
    return texts

async def get_dynamic_keywords(
    l1_scene: str, l2_fixtures: list, kanops_categories: list,
    ocr_texts: list, store_context: str = "Unknown UK Supermarket",
    semaphore: asyncio.Semaphore = None
) -> list:
    """
    Call Gemini Flash to generate context-aware product keywords.
    Manual retry logic:
      - 503 UNAVAILABLE: up to 8 attempts, exponential backoff 4->8->16->32->60->60->60->60s
      - 429 RESOURCE_EXHAUSTED: up to 6 attempts, fixed 30s wait
      - Other errors: raise immediately (don't waste retries)
    """
    ocr_str = ", ".join([t['text'] for t in ocr_texts]) if ocr_texts else "No readable text"

    prompt = f"""
    You are a grocery retail merchandising expert for '{store_context}'.
    Given the following store context and VISUALLY EXTRACTED TEXT (OCR), generate a list of 10-20 specific product names 
    that would likely be visible on these fixtures.

    Scene zone: {l1_scene}
    Fixtures present: {', '.join(l2_fixtures)}
    Known product categories (Kanops): {', '.join(kanops_categories)}
    
    VISUAL TEXT EXTRACTED FROM IMAGE (OCR): {ocr_str}

    CRITICAL RULES: 
    1. DO NOT guess random brands just because of the store_context. Strictly rely on the VISUAL TEXT EXTRACTED to infer specific products and brand names.
    2. DO NOT include packaging sizes, volumes (e.g., 500ml, 2L, multipack, gm). Output ONLY the core base product brand name (e.g., 'Coca-Cola Original Taste', 'Coca-Cola Zero Sugar').
    3. Return specific, visually identifiable product names matching the actual texts (Do NOT mix competitor brands).
    4. If OCR text says "No readable text", return at least 5 generic product names based on the scene/fixture type.
    Return strict JSON matching the schema.
    """

    max_503_attempts = 8
    max_429_attempts = 6
    attempt_503 = 0
    attempt_429 = 0

    async def _call():
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DynamicProductKeywords,
                temperature=0.3,
            ),
        )
        if not response.text:
            raise Exception("Empty response from Gemini")
        result = DynamicProductKeywords.model_validate_json(response.text)
        return result.keywords

    while True:
        try:
            if semaphore:
                async with semaphore:
                    return await _call()
            else:
                return await _call()

        except Exception as e:
            err_str = str(e)
            is_503 = "503" in err_str or "UNAVAILABLE" in err_str
            is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()

            if is_503:
                attempt_503 += 1
                if attempt_503 > max_503_attempts:
                    raise
                wait_s = min(4 * (2 ** (attempt_503 - 1)), 60)
                safe_print(f"    [503] Gemini overloaded. Waiting {wait_s}s "
                      f"(attempt {attempt_503}/{max_503_attempts})...")
                await asyncio.sleep(wait_s)

            elif is_429:
                attempt_429 += 1
                if attempt_429 > max_429_attempts:
                    raise
                safe_print(f"    [429] Rate limit hit. Waiting 30s "
                      f"(attempt {attempt_429}/{max_429_attempts})...")
                await asyncio.sleep(30)

            else:
                raise  # Non-transient error - skip immediately

def clip_l3_product_tag(image_path: str, keywords: list, clip_model, preprocess, device):
    """Run CLIP zero-shot against dynamic keywords to produce L3 tags."""
    import clip
    from PIL import Image

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    prompts = [f"A photo of {kw}" for kw in keywords]
    text = clip.tokenize(prompts, truncate=True).to(device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    tagged = []
    for kw, prob in zip(keywords, probs):
        tag_type = "Hard" if prob > 0.15 else "Soft" if prob > 0.05 else "Absence"
        tagged.append({
            "product": kw,
            "confidence": round(float(prob), 4),
            "tag_type": tag_type,
        })
    tagged.sort(key=lambda x: x["confidence"], reverse=True)
    return tagged


async def process_single_image(
    item: dict,
    kanops_categories: list,
    metadata_map: dict,
    clip_model,
    preprocess,
    device: str,
    semaphore: asyncio.Semaphore,
    cache: dict,
    lock: asyncio.Lock,
) -> dict | None:
    """
    Process one image: OCR -> Gemini keywords -> CLIP tags.
    Returns a result dict, or None if already cached / on error.
    """
    import clip
    img_path = item["image_path"]

    # Thread-safe check against shared cache
    async with lock:
        if img_path in cache:
            return None  # Already processed

    safe_print(f"\n--- L3 Tagging: {os.path.basename(img_path)} ---")
    try:
        l1_scene = item["L1"]["predicted_scene"]
        l2_fixtures = item["L2"]["fixtures_detected"]

        ocr_texts = extract_ocr_text(img_path)
        safe_print(f"  OCR extracted {len(ocr_texts)} text blocks")

        meta_info = metadata_map.get(img_path, {})
        store_ctx = meta_info.get("retailer_metadata", "Unknown")

        # Gemini API call with semaphore to cap concurrency
        keywords = await get_dynamic_keywords(
            l1_scene, l2_fixtures, kanops_categories, ocr_texts,
            store_context=store_ctx, semaphore=semaphore
        )

        safe_print(f"  Gemini returned {len(keywords)} keywords: {str(keywords[:5])[:120]}...")

        # If Gemini returned 0 keywords, skip CLIP tagging entirely.
        # l3_source="empty" signals the captioning stage NOT to use L3 constraints
        # (avoids hallucinating products that aren't visually confirmed).
        if not keywords:
            safe_print(f"  [SKIP] 0 keywords — no CLIP tagging. Marking as empty (no hallucination risk).")
            result = {
                "image_path": img_path,
                "L1": item["L1"],
                "L2": item["L2"],
                "L3_dynamic_keywords": [],
                "L3_product_tags": [],
                "ocr_text": ocr_texts,
                "l3_source": "empty"
            }
            return result

        l3_tags = clip_l3_product_tag(img_path, keywords, clip_model, preprocess, device)

        result = {
            "image_path": img_path,
            "L1": item["L1"],
            "L2": item["L2"],
            "L3_dynamic_keywords": keywords,
            "L3_product_tags": l3_tags,
            "ocr_text": ocr_texts,
            "l3_source": "gemini"
        }
        return result

    except Exception as e:
        err_str = str(e)
        is_transient = "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str
        safe_print(f"  [ERROR] {os.path.basename(img_path)}: {type(e).__name__} - skipping.")
        log_error(img_path, "l3_tagging", e)
        if is_transient:
            safe_print("  [COOLDOWN] API still struggling - pausing 15s before next image...")
            await asyncio.sleep(15)
        return None


async def run_l3_tagging():
    with open(IN_PATH, "r") as f:
        l1l2_results = json.load(f)
    with open("data/cache/kanops_mappings.json", "r") as f:
        kanops_data = json.load(f)

    try:
        with open("data/cache/metadata_mapped.json", "r") as f:
            metadata_map = json.load(f)
    except FileNotFoundError:
        metadata_map = {}

    kanops_categories = [m["kanops_subcategory"] for m in kanops_data["mappings"]]

    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    safe_print(f"[INFO] Using device: {device}")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # ── Checkpoint ──
    cache = load_existing_cache()
    already_done = len(cache)
    if already_done > 0:
        safe_print(f"[CHECKPOINT] {already_done} images already have L3 tags. Resuming...")

    # Filter to only unprocessed items
    todo = [item for item in l1l2_results if item["image_path"] not in cache]
    safe_print(f"[INFO] {len(todo)} images remaining to process (CONCURRENCY={CONCURRENCY})")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    new_count = 0

    # Process in batches to allow periodic auto-save
    batch_size = CONCURRENCY * 10  # e.g., 50 if CONCURRENCY=5
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]

        tasks = [
            process_single_image(
                item, kanops_categories, metadata_map,
                clip_model, preprocess, device,
                semaphore, cache, lock
            )
            for item in batch
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None:
                async with lock:
                    cache[result["image_path"]] = result
                new_count += 1

        # Auto-save after each batch
        save_cache(cache)
        total_done = already_done + new_count
        safe_print(f"  [AUTO-SAVE] Batch complete. New this session: {new_count} | Total: {total_done}/{len(l1l2_results)}")

    # Final save
    save_cache(cache)
    safe_print(f"\nSaved combined L1+L2+L3 results to {OUT_PATH}")
    safe_print(f"Processed this session: {new_count} | Total in cache: {len(cache)}")

if __name__ == "__main__":
    # Explicit event loop management to prevent aiohttp connector __del__
    # errors from causing a non-zero exit code (same fix as pipeline_4).
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_l3_tagging())
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    sys.exit(0)
