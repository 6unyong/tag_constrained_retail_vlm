import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def download_subset():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not found in .env file. Download may fail if the dataset is private/gated.")
        
    num_images = 2814 # Changed to match the exact thesis dataset size
    local_dir = "data/raw/train"
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Streaming the first {num_images} images from Hugging Face...")
    print("This will NOT download the entire 10K dataset archive!")
    
    try:
        # Load dataset in streaming mode so it doesn't download the whole thing
        dataset = load_dataset(
            "dresserman/kanops-open-retail-imagery", 
            split="train", 
            streaming=True,
            token=hf_token
        )
        
        count = 0
        for item in dataset:
            if count >= num_images:
                break
                
            # 'item' usually has 'image' as a PIL object and some identifier
            image = item.get("image")
            
            # Attempt to get a filename, otherwise generate one
            filename = item.get("image_id") or item.get("filename") or f"kanops_sample_{count:04d}.jpg"
                
            # Ensure it ends with .jpg as the pipeline expects
            if not str(filename).lower().endswith(('.png', '.jpg', '.jpeg')):
                filename = f"{filename}.jpg"
                
            # Just in case filename contains subdirectories
            filename = os.path.basename(str(filename))
            filepath = os.path.join(local_dir, filename)
            
            # Save the PIL image to disk
            if image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(filepath, format="JPEG", quality=95)
                count += 1
                if count % 10 == 0:
                    print(f"  -> Saved {count} / {num_images} images")
            
        print(f"\nSuccessfully saved {count} images to '{local_dir}'!")
        
    except Exception as e:
        print(f"\nError downloading subset: {e}")

if __name__ == "__main__":
    download_subset()
