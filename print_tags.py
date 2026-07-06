import json
import os
with open('data/cache/preview_l3_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for item in data:
    path = os.path.basename(item['image_path'])
    tags = item.get('L3_product_tags', [])
    print(f'\n--- {path} ---')
    for t in tags[:5]:
        print(f'  [CLIP {t["confidence"]}] {t["product"]}')
