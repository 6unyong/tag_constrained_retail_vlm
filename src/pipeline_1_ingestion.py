import os
import cv2
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from typing import Tuple

# Configuration
DATASET_URI = "hf://datasets/dresserman/kanops-open-retail-imagery/train"
METADATA_URI = "hf://datasets/dresserman/kanops-open-retail-imagery/metadata.csv"
OUTPUT_DIR = "data/processed"
BLUR_THRESHOLD = 50.0  # Adjustable Laplacian variance threshold
META_OUT_PATH = "data/cache/metadata_mapped.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

def is_blurry(image: Image.Image, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """Check if an image is blurry using the variance of the Laplacian."""
    open_cv_image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def load_existing_metadata() -> dict:
    """Load previously saved metadata to enable session resumption."""
    if os.path.exists(META_OUT_PATH):
        with open(META_OUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metadata(metadata_map: dict):
    """Persist current metadata map to disk."""
    with open(META_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_map, f, indent=4)

def ingest_data(limit: int = 0):
    """
    Ingest images from the HuggingFace dataset stream.
    
    Args:
        limit: Max number of images to process. 0 = no limit (full dataset).
    """
    print(f"Loading Hugging Face Dataset from dresserman/kanops-open-retail-imagery (Streaming)...")
    try:
        ds = load_dataset("dresserman/kanops-open-retail-imagery", split="train", streaming=True)
        print("Successfully connected to dataset stream!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"\nLoading Metadata CSV from {METADATA_URI}...")
    try:
        meta = pd.read_csv(METADATA_URI)
        print(f"Successfully loaded metadata! Total rows: {len(meta)}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # --- Checkpoint: Load previously processed metadata ---
    metadata_map = load_existing_metadata()
    already_done = len(metadata_map)
    if already_done > 0:
        print(f"\n[CHECKPOINT] Resuming: {already_done} images already processed. Skipping them.")

    stream = itertools.islice(ds, limit) if limit > 0 else ds
    limit_str = str(limit) if limit > 0 else "ALL"
    print(f"\nProcessing up to {limit_str} images...")

    processed_count = 0
    for i, item in enumerate(stream):
        img = item["image"]

        # Build the expected save path to check if already done
        save_path = os.path.join(OUTPUT_DIR, f"sample_{i}.jpg")
        # Normalise path separators for cross-platform key matching
        save_path_norm = save_path.replace("\\", "/")

        if save_path in metadata_map or save_path_norm in metadata_map:
            continue  # Already processed - skip without re-saving

        # --- Per-image try/except: one bad image won't kill the whole run ---
        try:
            if i >= len(meta):
                print(f"  [WARN] Image {i}: No metadata row available, skipping.")
                continue

            row = meta.iloc[i]

            blurry, score = is_blurry(img)
            status = "REJECTED (Blurry)" if blurry else "ACCEPTED (Sharp)"

            retailer = row.get("retailer", np.nan)
            store_name = row.get("store_name", np.nan)
            file_name = row.get("file_name", "")

            path_retailer = "Unknown"
            if isinstance(file_name, str) and "/" in file_name:
                parts = file_name.split("/")
                if len(parts) >= 3:
                    path_retailer = parts[2]

            val_retailer = str(retailer) if pd.notna(retailer) else "Unknown"
            val_store = str(store_name) if pd.notna(store_name) else "Unknown"

            if val_retailer != "Unknown":
                actual_retailer = val_retailer
            elif val_store != "Unknown":
                actual_retailer = val_store
            else:
                actual_retailer = path_retailer

            global_context = []
            if isinstance(file_name, str):
                if "2014" in file_name: global_context.append("Year 2014")
                if "halloween" in file_name.lower(): global_context.append("Halloween Theme")

            img.save(save_path)

            metadata_map[save_path] = {
                "image_id": int(row["image_id"]) if "image_id" in row else i,
                "retailer_metadata": actual_retailer,
                "global_context": global_context,
                "blur_score": round(score, 2),
                "status": status
            }

            ctx_str = f" | Context: {global_context}" if global_context else ""
            print(f"Image {i}: Laplacian = {score:.2f} [{status}] | Retailer: {actual_retailer}{ctx_str}")
            processed_count += 1

            # Auto-save every 50 images to prevent data loss on crash
            if processed_count % 50 == 0:
                save_metadata(metadata_map)
                print(f"  [AUTO-SAVE] Checkpoint saved at {processed_count} new images.")

        except Exception as e:
            print(f"  [ERROR] Image {i} ({save_path}): {e} — skipping.")
            continue

    # Final save
    save_metadata(metadata_map)
    print(f"\nSaved metadata mapping to {META_OUT_PATH}")
    print(f"Total processed this session: {processed_count} | Total in cache: {len(metadata_map)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest retail images from HuggingFace dataset.")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of images to process (0 = full dataset, default: 0)"
    )
    args = parser.parse_args()
    ingest_data(limit=args.limit)
