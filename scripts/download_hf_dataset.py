import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def download_kanops_data():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("=========================================================")
        print("WARNING: 'HF_TOKEN' not found in environment variables.")
        print("If this dataset is gated or private, the download will fail.")
        print("Please add HF_TOKEN=your_token_here to your .env file.")
        print("=========================================================\n")
        
    print("Downloading all 10,686 images from Kanops Open Retail Imagery dataset...")
    print("Streaming dataset to bypass Windows folder naming constraints...")
    
    local_dir = "data/raw/train"
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Load dataset in streaming mode
        dataset = load_dataset(
            "dresserman/kanops-open-retail-imagery", 
            split="train", 
            streaming=True,
            token=hf_token
        )
        
        count = 0
        for item in dataset:
            image = item.get("image")
            
            # Use image_id or generate a unique sequential name to avoid trailing space folder issues
            filename = item.get("image_id") or item.get("filename") or f"kanops_{count:05d}.jpg"
            
            if not str(filename).lower().endswith(('.png', '.jpg', '.jpeg')):
                filename = f"{filename}.jpg"
                
            # Flatten the directory structure to avoid Windows folder crash
            filename = os.path.basename(str(filename))
            filepath = os.path.join(local_dir, filename)
            
            if image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(filepath, format="JPEG", quality=95)
                count += 1
                if count % 100 == 0:
                    print(f"  -> Saved {count} images...")
                    
        print(f"\nDownload complete! All {count} images are securely saved to: {local_dir}")
        print("You can now run 'python src/pipeline_1_ingestion.py' to begin processing.")
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\n[해결 방법]")
        print("1. Hugging Face 홈페이지에서 해당 데이터셋의 접근 약관(Terms)에 동의하셨나요?")
        print("2. 프로젝트 최상단의 '.env' 파일에 정확한 'HF_TOKEN=hf_...' 값을 입력하셨나요?")

if __name__ == "__main__":
    download_kanops_data()
