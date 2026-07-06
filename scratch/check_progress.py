import json, os

cache_dir = "data/cache"
files = [
    "metadata_mapped.json",
    "corpus_induction_results.json",
    "kanops_mappings.json",
    "l1_l2_tag_results.json",
    "l1_l2_l3_tag_results.json",
    "hierarchical_tags_final.json",
    "clustered_routes.json",
    "final_captions.json",
]

print("=== Pipeline Progress Check ===")
for f in files:
    path = os.path.join(cache_dir, f)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        count = len(data) if isinstance(data, (list, dict)) else "?"
        print(f"  [OK] {f}: {count} entries")
    else:
        print(f"  [--] {f}: NOT FOUND")

# Also count processed images
proc_dir = "data/processed"
if os.path.exists(proc_dir):
    imgs = [f for f in os.listdir(proc_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    print(f"\n  data/processed/: {len(imgs)} images")
