import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("Loading metadata...")
metadata_path = hf_hub_download(
    repo_id="dresserman/kanops-open-retail-imagery", 
    filename="metadata.csv", 
    repo_type="dataset", 
    token=hf_token
)
meta = pd.read_csv(metadata_path)

print("Streaming dataset...")
ds = load_dataset("dresserman/kanops-open-retail-imagery", split="train", streaming=True, token=hf_token)

local_dir = "data/raw/train"
os.makedirs(local_dir, exist_ok=True)

downloaded = 0
skipped = 0

for i, item in enumerate(ds):
    if i >= len(meta):
        break
        
    row = meta.iloc[i]
    file_name_raw = str(row.get("file_name", f"image_{i}.jpg"))
    
    # Fix Windows invalid folder names (trailing spaces)
    parts = file_name_raw.split("/")
    cleaned_parts = [p.strip() for p in parts]
    cleaned_file_name = "/".join(cleaned_parts)
    
    # We want to save inside data/raw/train.
    # If the original path was train/2014/Aldi/img.jpg, we strip the first 'train/' to avoid train/train/train
    if cleaned_file_name.startswith("train/"):
        cleaned_file_name = cleaned_file_name[6:]
        
    filepath = os.path.join(local_dir, cleaned_file_name).replace("\\", "/")
    
    if os.path.exists(filepath):
        skipped += 1
        continue
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    image = item.get("image")
    if image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(filepath, format="JPEG", quality=95)
        downloaded += 1
        
        if downloaded % 50 == 0:
            print(f"Downloaded {downloaded} new images... (Skipped {skipped})")

print(f"Done! Downloaded {downloaded} new images. Skipped {skipped} existing images.")
