import json
import os
import asyncio
from src.pipeline_3b_l3_product_tagging import get_dynamic_keywords, clip_l3_product_tag, extract_ocr_text
from src.pipeline_5_mop_captioning import build_mop_prompt
import clip
import torch
from google import genai
from google.genai import types

# Setup models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

async def test_5_captions():
    # Load the 5 images from evaluation_data
    with open('data/eval_results/evaluation_data.json', 'r', encoding='utf-8') as f:
        eval_data = json.load(f)[:5]

    # Load ontology results to get L1 and L2
    with open('data/cache/corpus_induction_results.json', 'r', encoding='utf-8') as f:
        ontology = {os.path.basename(item['image_file']): item for item in json.load(f)}

    artifact_lines = [
        "# 5장 샘플 캡션 비교 리포트",
        "고객님께서 요청하신 5장의 이미지에 대해, **과거 GS1(베이스라인)**, **Kanops 버그**, **Option B(시각 교차 검증)** 3가지 버전의 최종 캡션을 비교했습니다.\n"
    ]

    for item in eval_data:
        filename = item['image_filename']
        image_path = os.path.abspath(f"data/processed/{filename}")
        
        # 1. Run Pipeline 3b (Option B)
        ont_item = ontology.get(filename)
        if not ont_item:
            print(f"Skipping {filename}: no ontology data")
            continue
            
        l1_scene = ont_item.get("scene_description", "retail shelf")
        l2_fixtures = ont_item.get("fixtures_found", [])
        
        print(f"Processing {filename}...")
        ocr_texts = extract_ocr_text(image_path)
        
        try:
            keywords = await get_dynamic_keywords(image_path, l1_scene, l2_fixtures, ocr_texts)
            if keywords:
                tagged = clip_l3_product_tag(image_path, keywords, clip_model, preprocess, device)
            else:
                tagged = []
        except Exception as e:
            print(f"Error on {filename}: {e}")
            tagged = []
        
        # Construct item dict for build_mop_prompt
        item_for_prompt = {
            "L1_scene": {"predicted_scene": l1_scene},
            "L2_fixtures": {"fixtures_detected": l2_fixtures},
            "L3_product_tags": tagged,
            "L4_attributes": {"operational_state": {}}
        }
        
        # 2. Run Pipeline 5 to generate caption
        print(f"Generating caption for {filename}...")
        prompt = build_mop_prompt(item_for_prompt, {})
        
        try:
            from PIL import Image
            img = Image.open(image_path)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img, prompt],
                config=types.GenerateContentConfig(temperature=0.3)
            )
            option_b_caption = response.text.strip()
        except Exception as e:
            option_b_caption = f"Error generating caption: {e}"

        # 3. Assemble artifact
        artifact_lines.extend([
            f"## {filename}",
            f"![{filename}](file:///{image_path.replace(chr(92), '/')})",
            "### 1. 과거 버전 (GS1 텍스트 앵커링)",
            f"> {item['baseline_caption']}\n",
            "### 2. 현재 버전 (Kanops 추상 버그)",
            f"> {item['mop_caption']}\n",
            "### 3. Option B (시각적 교차 검증 - 신규)",
            f"> **{option_b_caption}**\n",
            "---\n"
        ])

    # Write artifact
    artifact_path = "c:/Users/user/.gemini/antigravity/brain/40db0fc1-9baf-4af8-be51-a466939867aa/caption_comparison_preview.md"
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("\n".join(artifact_lines))
    print(f"Artifact created at {artifact_path}")

if __name__ == "__main__":
    asyncio.run(test_5_captions())
