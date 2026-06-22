"""
Task 11: L4 Attribute Tagging
- PaddleOCR: Extract visible text (price tags, promotions, brand names)
- CLIP Zero-shot: Operational attributes (stock level, cleanliness, promotion status)
- Combines everything into final hierarchical tag JSON (L1+L2+L3+L4)

10K Resilience features:
- Dict-based checkpoint: already-processed images skipped on re-run
- Per-image try/except: fault-tolerant — one bad image won't kill the whole run
- torch.cuda.empty_cache() after each image to prevent VRAM creep
- Auto-save every 50 images
"""
import os
import json
import torch
import clip
from PIL import Image

IN_PATH = "data/cache/l1_l2_l3_tag_results.json"
OUT_PATH = "data/cache/hierarchical_tags_final.json"
ERROR_LOG = "data/cache/error_log.txt"
AUTOSAVE_INTERVAL = 50

# --- Configuration ---
# You can experiment with different thresholds by setting the ABSENCE_THRESHOLD environment variable.
ABSENCE_THRESHOLD = float(os.environ.get("ABSENCE_THRESHOLD", 0.60))

# ── L4 Attribute labels for CLIP zero-shot (Conservative Tuning) ──
ATTRIBUTE_LABELS = {
    "stock_level": [
        "A photo of a tightly packed grocery shelf with absolutely no empty gaps or missing items",
        "A photo of a grocery shelf with obvious missing items and visible empty gaps",
        "A photo of a completely empty retail shelf with no products"
    ],
    "tidiness": [
        "A photo of a perfectly neat and organized retail display without any messy items",
        "A photo of a messy, disorganized, and untidy retail display with products out of place"
    ],
    "promotion": [
        "A photo of a retail shelf containing highly visible promotional signage or bright sale tags",
        "A photo of a standard retail shelf with no promotional signs or sale tags"
    ]
}

def log_error(img_path: str, error: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[pipeline_3c] {img_path} | {type(error).__name__}: {error}\n")

def load_existing_cache() -> dict:
    """Load previously saved L4 results as {image_path: result} dict."""
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {item["image_path"]: item for item in existing}
    return {}

def save_cache(cache: dict):
    os.makedirs("data/cache", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(cache.values()), f, indent=4, ensure_ascii=False)

def classify_attributes(image_path: str, clip_model, preprocess, device) -> dict:
    """Use CLIP zero-shot to classify operational attributes."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    results = {}

    with torch.no_grad():
        for attr_key, labels in ATTRIBUTE_LABELS.items():
            text_tokens = clip.tokenize(labels).to(device)
            logits_per_image, _ = clip_model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            max_idx = probs.argmax()
            max_prob = probs[max_idx]

            # Explicit Absence Encoding
            if max_prob < ABSENCE_THRESHOLD:
                final_label = "Ambiguous (Cannot be clearly determined from this image)"
            else:
                final_label = labels[max_idx].replace("A photo of a ", "").replace("A photo of an ", "")

            results[attr_key] = {
                "label": final_label,
                "confidence": round(float(max_prob), 3)
            }

    return results

def run_l4_tagging():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        prev_results = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    try:
        with open("data/cache/metadata_mapped.json", "r") as f:
            meta_map = json.load(f)
    except FileNotFoundError:
        meta_map = {}

    # --- Checkpoint: skip already-processed images ---
    cache = load_existing_cache()
    already_done = len(cache)
    if already_done > 0:
        print(f"[CHECKPOINT] {already_done} images already have L4 tags. Resuming...")

    new_count = 0
    for item in prev_results:
        img_path = item["image_path"]

        if img_path in cache:
            continue  # Already done — skip

        print(f"\n--- L4 Tagging: {os.path.basename(img_path)} ---")

        try:
            global_ctx = meta_map.get(img_path, {}).get("global_context", [])
            ocr_texts = item.get("ocr_text", [])
            print(f"  OCR (cached) found {len(ocr_texts)} text regions: {[t['text'] for t in ocr_texts[:5]]}")

            attributes = classify_attributes(img_path, clip_model, preprocess, device)
            print(f"  Attributes: {attributes}")

            final_tag = {
                "image_file": os.path.basename(img_path),
                "image_path": img_path,
                "global_context": global_ctx,
                "L1_scene": item["L1"],
                "L2_fixtures": item["L2"],
                "L3_products": {
                    "dynamic_keywords_used": item["L3_dynamic_keywords"],
                    "top_products": item["L3_product_tags"],  # All products (no cap)
                },
                "L4_attributes": {
                    "ocr_text": ocr_texts,
                    "operational_state": attributes,
                },
            }

            cache[img_path] = final_tag
            new_count += 1

            # Prevent VRAM accumulation over long runs
            if device == "cuda":
                torch.cuda.empty_cache()

            # Auto-save every AUTOSAVE_INTERVAL images
            if new_count % AUTOSAVE_INTERVAL == 0:
                save_cache(cache)
                print(f"  [AUTO-SAVE] Checkpoint at {new_count} new images (total: {len(cache)}).")

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: {e} — skipping.")
            log_error(img_path, e)
            continue

    # Final save
    save_cache(cache)
    print(f"\n=== Final hierarchical tags saved to {OUT_PATH} ===")
    print(f"Processed this session: {new_count} | Total in cache: {len(cache)}")

if __name__ == "__main__":
    run_l4_tagging()
