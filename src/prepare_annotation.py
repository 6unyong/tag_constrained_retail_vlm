import json
import random
import csv

MOP_FILE = "data/cache/final_captions_ollama.json"
BASE_FILE = "data/cache/final_captions_base.json"
OUT_FILE = "data/eval_results/evaluation_data.json"

def main():
    # 1. Load the 100 newly generated MOP captions from the fixed architecture
    with open(MOP_FILE, "r", encoding="utf-8") as f:
        mop_data = json.load(f)
        
    # 2. Load the LLaVA 3.8 Baseline captions
    with open(BASE_FILE, "r", encoding="utf-8") as f:
        base_data = json.load(f)
        
    # Build a lookup for baseline captions
    base_map = {item['image_file']: item['FINAL_CAPTION'] for item in base_data}
    
    final_set = []
    
    for i, item in enumerate(mop_data):
        image_file = item["image_file"]
        mop_caption = item["FINAL_CAPTION"]
        baseline_caption = base_map.get(image_file, "Baseline caption not found")
        
        final_set.append({
            "id": i + 1,
            "image_filename": image_file,
            "baseline_caption": baseline_caption,
            "mop_caption": mop_caption,
            "llm_preference": "N/A", # Not run through judge yet for the new architecture
            "group": "Fixed Architecture (LLaVA)",
            "annotations": {} # Initialize empty dict for Annotator 1 / Annotator 2
        })
        
    # Shuffle the final order so the human doesn't have a predictable order
    random.seed(42)
    random.shuffle(final_set)
    
    # Re-assign sequential IDs
    for i, item in enumerate(final_set):
        item["id"] = i + 1
        
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_set, f, ensure_ascii=False, indent=4)
        
    # Also output to CSV for easy Google Forms/Sheets import
    csv_out_file = OUT_FILE.replace(".json", ".csv")
    with open(csv_out_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Image Filename", "Baseline Caption", "MOP Caption", "LLM Preference", "Group", "Hallucination in Baseline (O/X)", "Hallucination in MOP (O/X)", "Annotator Preference"])
        for item in final_set:
            writer.writerow([
                item["id"], 
                item["image_filename"], 
                item["baseline_caption"], 
                item["mop_caption"], 
                item["llm_preference"], 
                item["group"],
                "", "", ""
            ])
            
    print(f"Generated {len(final_set)} samples for annotation from the fixed LLaVA architecture.")
    print(f"Saved to {OUT_FILE} and {csv_out_file}")

if __name__ == "__main__":
    main()
