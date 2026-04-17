"""
Ablation Baseline: Tests pure LLaVA 1.5 7B without the MOP pipeline.
Generates vanilla captions for the same image set used by the MOP pipeline,
enabling a fair quantitative comparison (MOP vs Baseline).

10K Resilience features:
- Dynamic file discovery: processes all images in data/processed/ (no hardcoding)
- Dict-based checkpoint: already-captioned images are skipped on re-run
- Per-image try/except with error logging
- Auto-save every 50 images
"""
import os
import json
import base64
import requests
import argparse
from glob import glob

OUT_PATH = "data/cache/baseline_captions.json"
SAMPLE_LIST_PATH = "data/cache/sample_image_list.json"  # written by pipeline_5 --sample
ERROR_LOG = "data/cache/error_log.txt"
AUTOSAVE_INTERVAL = 50
OLLAMA_TIMEOUT = 120  # seconds — llava needs up to 60-90s for vision tasks

# The vanilla baseline prompt — deliberately simple and unconstrained
# to expose the full extent of LLaVA's unguided hallucination behaviour.
BASELINE_PROMPT = (
    "Describe the state of this grocery retail image in 2 sentences. "
    "Notice the products, fixtures, text signs, and overall operational condition."
)


def log_error(img_path: str, error: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[ablation_baseline] {img_path} | {type(error).__name__}: {error}\n")


def load_existing_cache() -> dict:
    """Load previously generated baseline captions as {image_path: item} dict."""
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {item["image_path"]: item for item in existing}
    return {}


def save_cache(cache: dict):
    os.makedirs("data/cache", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(cache.values()), f, indent=4, ensure_ascii=False)


def get_base64_img(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_baseline_captions(from_mop: bool = False, sample: int = None):
    # Determine image list
    if from_mop and os.path.exists(SAMPLE_LIST_PATH):
        with open(SAMPLE_LIST_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"[FROM-MOP] Using {len(samples)} images from MOP sample list.")
    elif from_mop:
        print(f"[WARN] --from-mop set but {SAMPLE_LIST_PATH} not found. "
              "Run pipeline_5_mop_captioning.py --sample N first.")
        return
    else:
        samples = sorted(glob("data/processed/*.jpg"))
        if sample:
            samples = samples[:sample]
            print(f"[SAMPLE] Using first {len(samples)} images.")

    if not samples:
        print("[WARN] No images found in data/processed/. Run pipeline_1 first.")
        return

    print(f"Found {len(samples)} images for baseline captioning.")

    # --- Checkpoint: skip already-captioned images ---
    cache = load_existing_cache()
    already_done = len(cache)
    if already_done > 0:
        print(f"[CHECKPOINT] {already_done} baseline captions already exist. Resuming...")

    new_count = 0
    for img_path in samples:
        if img_path in cache:
            continue  # Already captioned — skip Ollama call

        print(f"Generating baseline caption for {os.path.basename(img_path)}...")

        try:
            b64 = get_base64_img(img_path)

            payload = {
                "model": "llava:latest",
                "prompt": BASELINE_PROMPT,
                "images": [b64],
                "stream": False,
                "options": {"timeout": OLLAMA_TIMEOUT},
            }

            res = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )
            res.raise_for_status()
            txt = res.json().get("response", "").strip()

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: {e} - skipping.")
            log_error(img_path, e)
            txt = "GENERATION_ERROR"

        print(f"  Vanilla LLaVA: {txt[:120]}...")

        cache[img_path] = {
            "image_file": os.path.basename(img_path),
            "image_path": img_path,
            "FINAL_CAPTION": txt,
            # Empty tag stubs so pipeline_6/7 CHAIR eval works on baseline too
            "L1_scene": {},
            "L2_fixtures": {},
            "L3_products": {},
            "L4_attributes": {},
            "global_context": [],
        }
        new_count += 1

        if new_count % AUTOSAVE_INTERVAL == 0:
            save_cache(cache)
            print(f"  [AUTO-SAVE] Checkpoint at {new_count} new captions (total: {len(cache)}).")

    save_cache(cache)
    print(f"\nBaseline captions saved to {OUT_PATH}")
    print(f"Generated this session: {new_count} | Total in cache: {len(cache)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation baseline caption generator")
    parser.add_argument("--from-mop", action="store_true",
                        help="Process the same images as pipeline_5 --sample (reads sample_image_list.json)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process first N images from data/processed/ (if not using --from-mop)")
    args = parser.parse_args()
    generate_baseline_captions(from_mop=args.from_mop, sample=args.sample)
