import os
import urllib.request

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading {os.path.basename(target_path)}...")
        urllib.request.urlretrieve(url, target_path)
        print("Download complete!")
    else:
        print(f"{os.path.basename(target_path)} already exists.")

if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)
    
    # GroundingDINO SwinT Config
    config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    download_file(config_url, "weights/GroundingDINO_SwinT_OGC.py")
    
    # GroundingDINO SwinT Weights
    weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    download_file(weights_url, "weights/groundingdino_swint_ogc.pth")
    
    print("All weights setup successfully.")
