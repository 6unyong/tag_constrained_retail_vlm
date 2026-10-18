import os
import torch
import json
from glob import glob
from PIL import Image
import clip

from groundingdino.util.inference import load_model, load_image, predict

CACHE_PATH = "data/cache/l1_l2_tag_results.json"
ERROR_LOG_PATH = "data/cache/error_log.txt"
AUTOSAVE_INTERVAL = 50

def log_error(img_path: str, error: Exception):
    """Append a single-line error record to the error log file."""
    os.makedirs("data/cache", exist_ok=True)
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[pipeline_3] {img_path} | {type(error).__name__}: {error}\n")

def load_existing_cache() -> dict:
    """Load previously saved results as {image_path: result} dict for resumption."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {item["image_path"]: item for item in existing}
    return {}

def save_cache(cache: dict):
    """Persist the current cache dict as a JSON list."""
    os.makedirs("data/cache", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(list(cache.values()), f, indent=4)

def setup_models():
    print("Loading CLIP (OpenAI)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    print("Loading Grounding DINO...")
    config_path = "weights/GroundingDINO_SwinT_OGC.py"
    weight_path = "weights/groundingdino_swint_ogc.pth"

    if not os.path.exists(config_path) or not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing Grounding DINO weights. Expected at {weight_path}")

    dino_model = load_model(config_path, weight_path)
    dino_model = dino_model.to(device)

    return clip_model, preprocess, dino_model, device

def tag_l1_scene(image_path, clip_model, preprocess, device):
    """Task 9 - L1 Feature: Detect Scene (Soft vs Hard probability)"""
    scene_labels = [
        "A photo of a standard continuous grocery aisle shelf",
        "A photo of a promotional endcap at the end of an aisle",
        "A photo of a grocery checkout area or till",
        "A photo of a standalone promotional display bin"
    ]

    with Image.open(image_path) as pil_img:
        image = preprocess(pil_img).unsqueeze(0).to(device)
    text = clip.tokenize(scene_labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    best_idx = probs[0].argmax()
    best_label = scene_labels[best_idx].replace("A photo of a ", "")
    confidence = probs[0][best_idx]

    tag_type = "Hard" if confidence > 0.85 else "Soft" if confidence >= 0.60 else "Absence"

    return {
        "predicted_scene": best_label,
        "confidence": round(float(confidence), 3),
        "tag_type": tag_type
    }

def tag_l2_fixture(image_path, dino_model):
    """Task 9 - L2 Feature: Box Fixtures"""
    TEXT_PROMPT = "shelf . refrigerator . display box . display bin ."
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    image_source, image_tensor = load_image(image_path)

    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    return {
        "fixtures_detected": phrases,
        "num_fixtures": len(phrases),
        "avg_confidence": round(sum(logits.tolist()) / len(logits), 3) if len(logits) > 0 else 0.0
    }

if __name__ == "__main__":
    print("=== Pipeline 3: Stage 1 Tagging (L1 Scene + L2 Fixture) ===")

    # Load models once before the loop (expensive operation)
    clip_m, preproc, dino_m, dev = setup_models()

    test_images = sorted(glob("data/processed/*.jpg"))
    print(f"\nFound {len(test_images)} processed images.")

    # --- Checkpoint: load existing results, skip already-processed images ---
    cache = load_existing_cache()
    already_done = len(cache)
    if already_done > 0:
        print(f"[CHECKPOINT] {already_done} images already tagged. Resuming from where we left off.")

    new_count = 0
    for img in test_images:
        if img in cache:
            continue  # Already tagged — skip entirely (no GPU work, no re-save)

        print(f"\n--- Tagging Image: {os.path.basename(img)} ---")

        # Per-image try/except: one corrupt image won't kill the entire run
        try:
            l1_result = tag_l1_scene(img, clip_m, preproc, dev)
            print(f"  L1 (Scene): {l1_result}")

            l2_result = tag_l2_fixture(img, dino_m)
            print(f"  L2 (Fixture): {l2_result}")

            cache[img] = {
                "image_path": img,
                "L1": l1_result,
                "L2": l2_result
            }
            new_count += 1

            # Auto-save every AUTOSAVE_INTERVAL images
            if new_count % AUTOSAVE_INTERVAL == 0:
                save_cache(cache)
                print(f"  [AUTO-SAVE] Checkpoint at {new_count} new images.")

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img)}: {e} - skipping.")
            log_error(img, e)
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final save
    save_cache(cache)
    print(f"\nSaved tag results to {CACHE_PATH}")
    print(f"Processed this session: {new_count} | Total in cache: {len(cache)}")
