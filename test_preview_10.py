import asyncio
import json
import os
import sys

# Patch constants
import src.pipeline_3b_l3_product_tagging as p3b
p3b.IN_PATH = "data/cache/clustered_routes_preview.json"
p3b.OUT_PATH = "data/cache/preview_l3_results.json"

async def test_10():
    print("Running Pipeline 3b (Option B Vision) on 10 preview images...")
    await p3b.run_l3_tagging()
    
    # Check results
    with open(p3b.OUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("\n\n--- RESULTS ---")
    for item in data:
        img_file = item["image_file"]
        l3 = item.get("L3_products", {}).get("top_products", [])
        print(f"\n{img_file}:")
        for p in l3[:5]:
            print(f"  - {p['product']} ({p['confidence']}) [{p['tag_type']}]")

if __name__ == "__main__":
    asyncio.run(test_10())
