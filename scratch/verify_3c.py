import json
from collections import Counter

path = 'data/cache/hierarchical_tags_final.json'
try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total entries in final tags: {len(data)}")
    
    sample = data[0]
    print("\n--- Sample Structure Check ---")
    for key in ['L1_scene', 'L2_fixtures', 'L3_products', 'L4_attributes']:
        print(f"Has {key}? {key in sample}")
    
    if "L4_attributes" in sample and "operational_state" in sample["L4_attributes"]:
        l4 = sample["L4_attributes"]["operational_state"]
        print("\n--- L4 Attributes (Pipeline 3c) ---")
        for k, v in l4.items():
            print(f"{k}: {v.get('label')} (Conf: {v.get('confidence')})")

    valid_l3 = sum(1 for d in data if "L3_products" in d and len(d["L3_products"].get("product_keywords", [])) > 0)
    
    # Calculate average confidence of L4
    l4_conf_sum = 0
    l4_conf_count = 0
    
    for d in data:
        if "L4_attributes" in d and "operational_state" in d["L4_attributes"]:
            for k, v in d["L4_attributes"]["operational_state"].items():
                if isinstance(v, dict) and "confidence" in v:
                    l4_conf_sum += v["confidence"]
                    l4_conf_count += 1
                    
    avg_l4_conf = (l4_conf_sum / l4_conf_count) if l4_conf_count > 0 else 0

    print(f"\n--- Statistics ---")
    print(f"Images with at least 1 L3 Product Keyword: {valid_l3} / {len(data)}")
    print(f"Average L4 Attribute Confidence: {avg_l4_conf:.3f}")
    print(f"Total parsed L4 attributes: {l4_conf_count}")

except Exception as e:
    print(f"Verification failed: {e}")
