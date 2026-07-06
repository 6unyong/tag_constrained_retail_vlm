import json
import random

IN_FILE = "data/eval_results/llm_judge_scores_k2.json"
OUT_FILE = "data/eval_results/evaluation_data.json"

def main():
    with open(IN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    scores = data.get("detailed_scores", [])
    
    # Separate into all samples and baseline-preferred samples
    baseline_wins = [s for s in scores if s.get("preference") == "Baseline"]
    
    # We want 50 random samples (Group A)
    # We want 50 baseline-preferred samples (Group B)
    # To ensure no overlap (though not strictly necessary, it's cleaner),
    # let's pick 50 from the whole population first, then remove them from baseline_wins,
    # and pick 50 from the remaining baseline_wins.
    
    random.seed(42) # For reproducibility
    
    group_a = random.sample(scores, 50)
    
    # Remove group A from baseline_wins to avoid exact duplicate questions in annotation
    group_a_ids = {s['image_file'] for s in group_a}
    remaining_baseline_wins = [s for s in baseline_wins if s['image_file'] not in group_a_ids]
    
    group_b = random.sample(remaining_baseline_wins, 50)
    
    # Combine and shuffle
    final_set = []
    
    for i, s in enumerate(group_a):
        final_set.append({
            "id": i + 1,
            "image_filename": s["image_file"],
            "baseline_caption": s["base_caption"],
            "mop_caption": s["mop_caption"],
            "llm_preference": s["preference"],
            "group": "A (Random)",
            "annotations": None
        })
        
    for i, s in enumerate(group_b):
        final_set.append({
            "id": i + 51,
            "image_filename": s["image_file"],
            "baseline_caption": s["base_caption"],
            "mop_caption": s["mop_caption"],
            "llm_preference": s["preference"],
            "group": "B (LLM Disagreement)",
            "annotations": None
        })
        
    # Shuffle the final order so the human doesn't know which group they are in
    random.shuffle(final_set)
    
    # Re-assign sequential IDs
    for i, item in enumerate(final_set):
        item["id"] = i + 1
        
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_set, f, ensure_ascii=False, indent=4)
        
    print(f"Generated {len(final_set)} samples for annotation.")
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
