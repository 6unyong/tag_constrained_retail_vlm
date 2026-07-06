import json
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

with open('data/eval_results/evaluation_data.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

with open('data/cache/l1_l2_l3_tag_results.json', 'r', encoding='utf-8') as f:
    kanops_tags = json.load(f)

with open('data/cache/preview_l3_results.json', 'r', encoding='utf-8') as f:
    option_b_tags = json.load(f)

def get_baseline(filename):
    for item in eval_data:
        if item['image_filename'] == filename:
            return item.get('baseline_caption', '')
    return "N/A"

def get_kanops(filename):
    for item in kanops_tags:
        if item['image_path'].endswith(filename):
            return item.get('L3_product_tags', [])
    return []

def get_option_b(filename):
    for item in option_b_tags:
        if item['image_path'].endswith(filename):
            return item.get('L3_product_tags', [])
    return []

print("# 태그 성능 비교 리포트\n")
for item in option_b_tags:
    filename = os.path.basename(item['image_path'])
    print(f"## {filename}")
    
    # 1. GS1
    baseline = get_baseline(filename)
    # The GS1 tags can be inferred from the baseline caption, but let's just show the caption
    print(f"**1. 이전 버젼(GS1 텍스트)에서 캡션에 반영된 제품들:**\n{baseline[:150]}...\n")
    
    # 2. Kanops
    k_tags = get_kanops(filename)
    k_str = ", ".join([f"[{t['confidence']:.2f}] {t['product']}" for t in k_tags[:3]]) if k_tags else "None"
    print(f"**2. 현재 버젼(Kanops 버그) 태그:**\n{k_str}\n")
    
    # 3. Option B
    o_tags = get_option_b(filename)
    o_str = ", ".join([f"[{t['confidence']:.2f}] {t['product']}" for t in o_tags[:3]]) if o_tags else "None"
    print(f"**3. 수정 후(Option B 시각 모델) 태그:**\n{o_str}\n")
    print("-" * 40)
