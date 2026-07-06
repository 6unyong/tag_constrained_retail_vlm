import json

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return str(e)

print("=== PIPELINE 1: Metadata & Blur Filter ===")
m = load_json('data/cache/metadata_mapped.json')
if isinstance(m, list):
    print(f"Total entries: {len(m)}")
    print(f"Sample[0]: {json.dumps(m[0], indent=2, ensure_ascii=False)}")
else:
    print(f"Error: {m}")

print("\n=== PIPELINE 2: Corpus Induction (Gemini) ===")
ci = load_json('data/cache/corpus_induction_results.json')
if isinstance(ci, list):
    print(f"Total entries: {len(ci)}")
    print(f"Sample[0]: {json.dumps(ci[0], indent=2, ensure_ascii=False)}")
else:
    print(f"Error: {ci}")

print("\n=== PIPELINE 2b: Kanops Mapping ===")
k = load_json('data/cache/kanops_mappings.json')
if isinstance(k, list):
    print(f"Total entries: {len(k)}")
    print(f"Sample[0]: {json.dumps(k[0], indent=2, ensure_ascii=False)}")
else:
    print(f"Error: {k}")

print("\n=== PIPELINE 3: L1 Scene & L2 Fixture (CLIP + GroundingDINO) ===")
l3 = load_json('data/cache/l1_l2_tag_results.json')
if isinstance(l3, list):
    print(f"Total entries: {len(l3)}")
    print(f"Sample[0]: {json.dumps(l3[0], indent=2, ensure_ascii=False)}")
else:
    print(f"Error: {l3}")
