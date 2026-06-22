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
FFT_CACHE_PATH = "data/cache/fft_threshold.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

def get_fft_magnitude(image: Image.Image) -> float:
    """Calculate the high-frequency magnitude of an image using FFT."""
    open_cv_image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    r = 30  # radius for high pass filter
    fshift[crow - r:crow + r, ccol - r:ccol + r] = 0
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return float(np.mean(img_back))

def compute_dynamic_threshold(ds_stream, num_samples=100) -> float:
    print(f"Scanning first {num_samples} images to calculate dynamic FFT threshold...")
    magnitudes = []
    for i, item in enumerate(ds_stream):
        if i >= num_samples:
            break
        mag = get_fft_magnitude(item["image"])
        magnitudes.append(mag)
        if (i + 1) % 20 == 0:
            print(f"  Scanned {i+1}/{num_samples} images...")
            
    if not magnitudes:
        return 10.0 # fallback
        
    optimal_threshold = float(np.percentile(magnitudes, 10))
    print(f"Dynamic FFT Threshold calculated (10th percentile): {optimal_threshold:.2f}")
    return optimal_threshold

def is_blurry(image: Image.Image, fft_threshold: float, lap_threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float, float]:
    """Check if an image is blurry using a hybrid Laplacian and FFT filter."""
    open_cv_image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    
    lap_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    fft_magnitude = get_fft_magnitude(image)
    
    is_blur = (lap_variance < lap_threshold) or (fft_magnitude < fft_threshold)
    
    return is_blur, lap_variance, fft_magnitude

def load_existing_metadata() -> dict:
    if os.path.exists(META_OUT_PATH):
        with open(META_OUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metadata(metadata_map: dict):
    with open(META_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_map, f, indent=4)

def ingest_data(limit: int = 0):
    print(f"Loading Hugging Face Dataset from dresserman/kanops-open-retail-imagery (Streaming)...")
    try:
        ds = load_dataset("dresserman/kanops-open-retail-imagery", split="train", streaming=True)
        print("Successfully connected to dataset stream!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Dynamic FFT Thresholding
    if os.path.exists(FFT_CACHE_PATH):
        with open(FFT_CACHE_PATH, "r") as f:
            fft_threshold = json.load(f)["threshold"]
        print(f"Loaded cached optimal FFT threshold: {fft_threshold:.2f}")
    else:
        fft_threshold = compute_dynamic_threshold(ds, num_samples=100)
        with open(FFT_CACHE_PATH, "w") as f:
            json.dump({"threshold": fft_threshold}, f)
        # Re-initialize stream because we consumed the first 100
        ds = load_dataset("dresserman/kanops-open-retail-imagery", split="train", streaming=True)

    print(f"\nLoading Metadata CSV from {METADATA_URI}...")
    try:
        meta = pd.read_csv(METADATA_URI)
        print(f"Successfully loaded metadata! Total rows: {len(meta)}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

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

        save_path = os.path.join(OUTPUT_DIR, f"sample_{i}.jpg")
        save_path_norm = save_path.replace("\\", "/")

        if save_path in metadata_map or save_path_norm in metadata_map:
            continue

        try:
            if i >= len(meta):
                print(f"  [WARN] Image {i}: No metadata row available, skipping.")
                continue

            row = meta.iloc[i]

            blurry, lap_score, fft_score = is_blurry(img, fft_threshold)
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
                "blur_score": round(lap_score, 2),
                "fft_score": round(fft_score, 2),
                "status": status
            }

            ctx_str = f" | Context: {global_context}" if global_context else ""
            print(f"Image {i}: Laplacian={lap_score:.1f}, FFT={fft_score:.1f} [{status}] | {actual_retailer}{ctx_str}")
            processed_count += 1

            if processed_count % 50 == 0:
                save_metadata(metadata_map)
                print(f"  [AUTO-SAVE] Checkpoint saved at {processed_count} new images.")

        except Exception as e:
            print(f"  [ERROR] Image {i} ({save_path}): {e} — skipping.")
            continue

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
