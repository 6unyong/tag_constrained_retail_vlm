import os
import cv2
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_blurry(image: Image.Image, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """Check if an image is blurry using the variance of the Laplacian."""
    # Convert PIL Image to cv2 format (numpy array)
    open_cv_image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def ingest_data():
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
        print("\nMetadata Head Preview:")
        print(meta.head())
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    print("\nProcessing a small sample (first 3 images) for Quality Filtering...")
    import itertools
    for i, item in enumerate(itertools.islice(ds, 3)):
        img = item["image"]
        
        blurry, score = is_blurry(img)
        status = "REJECTED (Blurry)" if blurry else "ACCEPTED (Sharp)"
        
        print(f"Image {i}: Laplacian Variance = {score:.2f} -> {status}")
        
        # Save sample
        save_path = os.path.join(OUTPUT_DIR, f"sample_{i}.jpg")
        img.save(save_path)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    ingest_data()
