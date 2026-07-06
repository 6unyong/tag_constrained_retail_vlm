import json
import os
import sys

sys.path.append(os.path.abspath('src'))
from pipeline_5_mop_captioning import build_mop_prompt

data = json.load(open('data/cache/clustered_routes_preview.json', encoding='utf-8'))

brain_dir = r'C:\Users\user\.gemini\antigravity\brain\40db0fc1-9baf-4af8-be51-a466939867aa'
out_path = os.path.join(brain_dir, 'prompt_diagnostics.md')

md = "# Tag & Prompt Diagnostics\n\n"

for i, item in enumerate(data[:2]):
    img_file = item['image_file']
    img_path = item['image_path']
    abs_img_path = os.path.join(brain_dir, img_file).replace('\\', '/')
    
    md += f"## Image {i+1}: {img_file}\n\n"
    md += f"![{img_file}](file:///{abs_img_path})\n\n"
    
    md += "### Extracted Tags\n"
    md += f"- **L1 Scene:** {item['L1_scene'].get('predicted_scene')}\n"
    md += f"- **L2 Fixtures:** {', '.join(item['L2_fixtures'].get('fixtures_detected', []))}\n"
    md += "- **L3 Products:**\n"
    for p in item.get('L3_products', {}).get('top_products', []):
        md += f"  - `{p.get('product')}` (Conf: {p.get('confidence')}, Type: {p.get('tag_type')})\n"
    md += f"- **L4 Ops:** `{json.dumps(item.get('L4_attributes', {}))}`\n\n"
    
    prompt = build_mop_prompt(item, {})
    md += "### Assembled MOP Prompt\n"
    md += f"```text\n{prompt}\n```\n\n"
    md += "---\n\n"

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(md)
