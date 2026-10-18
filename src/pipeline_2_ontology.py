import os
import sys
import asyncio
import json
import time
from glob import glob
from pydantic import BaseModel, Field
from typing import List

from src.utils.async_gemini import generate_structured_vision_async

# ── Constants ──
OUT_PATH = "data/cache/corpus_induction_results.json"
ERROR_LOG = "data/cache/error_log.txt"
AUTOSAVE_INTERVAL = 50
CONCURRENCY = 3

# 1. Define the Discovery Schema for Ontology (Corpus Induction Phase)
class CorpusInductionResult(BaseModel):
    image_file: str = Field(..., description="Name of the processed image file")
    scene_description: str = Field(..., description="Overall location context, e.g., ambient aisle, endcap, checkout")
    fixtures_found: List[str] = Field(..., description="List of physical structures holding products")
    product_categories: List[str] = Field(..., description="General grocery categories found, e.g., soft drinks, fresh produce")
    operational_issues: List[str] = Field(..., description="Any visible issues like empty shelves, messy displays, or promotional flags")


def log_error(img_path: str, error: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[pipeline_2] {img_path} | {type(error).__name__}: {error}\n")


def load_existing_cache() -> dict:
    """
    Load previously saved corpus induction results as {image_file: result} dict
    for checkpoint resumption.
    """
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {item["image_file"]: item for item in existing}
    return {}


def save_cache(cache: dict):
    os.makedirs("data/cache", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(cache.values()), f, indent=4, ensure_ascii=False)


async def process_image_for_induction(image_path: str, semaphore: asyncio.Semaphore) -> dict:
    prompt = """
    You are an expert in Retail Visual Merchandising and Grocery Store Operations.
    Please analyze this image and extract raw descriptive noun phrases (Corpus Induction).
    Focus on:
    1. The general scene/zone (e.g. ambient aisle, chiller, checkout).
    2. The physical fixtures present (e.g. gondola shelving, cardboard dump bin, promotional endcap).
    3. The product categories visible (broad Kanops parent level, like 'Soft Drinks and Mixers', 'Biscuits').
    4. Any operational states or issues (e.g. fully stocked, out of stock gap, messy, active promotion).
    Return strict JSON matching the schema.
    """
    print(f"Processing {image_path}...")
    try:
        async with semaphore:
            result = await generate_structured_vision_async(
                image_path=image_path,
                prompt=prompt,
                response_schema=CorpusInductionResult,
                model_name="gemini-2.5-pro"
            )
        # Ensure the filename is injected
        result.image_file = os.path.basename(image_path)
        return result.model_dump()
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        log_error(image_path, e)
        return {"image_file": os.path.basename(image_path), "error": str(e)}

async def run_corpus_induction():
    image_dir = "data/processed"
    images = glob(os.path.join(image_dir, "*.jpg"))
    
    if not images:
        print(f"No images found in {image_dir}.")
        return

    # In a full run, we would limit this to a stratified sample of 100 images
    print(f"Found {len(images)} images for Corpus Induction Testing.")

    # ── Checkpoint ──
    cache = load_existing_cache()
    already_done = len(cache)
    if already_done > 0:
        print(f"[CHECKPOINT] {already_done} images already processed. Resuming...")

    # Filter to only unprocessed images
    todo = [img for img in images if os.path.basename(img) not in cache]
    print(f"[INFO] {len(todo)} images remaining to process (CONCURRENCY={CONCURRENCY})")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    new_count = 0

    # Process in batches to allow periodic auto-save
    batch_size = CONCURRENCY * 5
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]

        tasks = [process_image_for_induction(img, semaphore) for img in batch]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None:
                img_file = result["image_file"]
                cache[img_file] = result
                new_count += 1

        # Auto-save after each batch
        save_cache(cache)
        total_done = already_done + new_count
        print(f"  [AUTO-SAVE] Batch complete. New this session: {new_count} | Total: {total_done}/{len(images)}")

    # Final save
    save_cache(cache)

    print(f"\nCorpus Induction complete! Results saved to {OUT_PATH}")
    print(f"Processed this session: {new_count} | Total in cache: {len(cache)}")
    results_list = list(cache.values())
    print("Preview of first result:")
    if results_list:
        print(json.dumps(results_list[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # Explicit event loop management to prevent aiohttp connector __del__
    # errors from causing a non-zero exit code (same fix as pipeline_3b).
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_corpus_induction())
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    sys.exit(0)
