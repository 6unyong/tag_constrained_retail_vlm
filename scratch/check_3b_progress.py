import json
import os

path = 'data/cache/l1_l2_l3_tag_results.json'
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Current progress: {len(data)} / 10686 images processed.")
else:
    print("File not found. Progress is 0 or saving hasn't happened yet.")
