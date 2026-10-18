"""
Task 10: L3 Product Tagging (100% Local Edge Version)
Step 1 — Send L1/L2 context + OCR to Ollama (llava-phi3) -> get dynamic product keyword list
Step 2 — Use those keywords as CLIP zero-shot prompts -> produce L3 probabilistic tags

100% Offline Edge Computing version substituting Gemini API with local VLM.
"""
import os
import sys
import json
import asyncio
import time
from typing import List
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
if sys.stderr.encoding != "utf-8":
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

import torch
from paddleocr import PaddleOCR
import ollama

load_dotenv()

ocr = PaddleOCR(use_textline_orientation=True, lang="en")

IN_PATH = "data/cache/l1_l2_tag_results.json"
OUT_PATH = "data/cache/l1_l2_l3_tag_results_ollama.json"
ERROR_LOG = "data/cache/error_log_ollama.txt"
AUTOSAVE_INTERVAL = 50
CONCURRENCY = 1 
DEFAULT_MODEL = "llava-phi3"

def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

def log_error(img_path: str, stage: str, error: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[pipeline_3b_ollama|{stage}] {img_path} | {type(error).__name__}: {error}\n")

def load_existing_cache() -> dict:
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {item["image_path"]: item for item in existing}
    return {}

def save_cache(cache: dict):
    os.makedirs("data/cache", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(cache.values()), f, indent=4, ensure_ascii=False)

def extract_ocr_text(image_path: str) -> list:
    try:
        result = ocr.ocr(image_path, cls=True)
    except Exception:
        return []
    texts = []
    if not result or not result[0]:
        return texts
    try:
        for line in result[0]:
            if line is None or len(line) < 2:
                continue
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:
                texts.append({"text": text, "confidence": round(float(confidence), 3)})
    except (TypeError, AttributeError):
        pass
    return texts

async def get_dynamic_keywords(
    image_path: str, l1_scene: str, l2_fixtures: list,
    ocr_texts: list, store_context: str = "Unknown UK Supermarket"
) -> list:
    ocr_str = ", ".join([t['text'] for t in ocr_texts]) if ocr_texts else "No readable text"

    prompt = f"""
    You are a grocery retail merchandising expert for '{store_context}'.
    Based on the image and the VISUALLY EXTRACTED TEXT (OCR) below, generate a list of 10-20 specific product brand names visible on these fixtures.

    Scene zone: {l1_scene}
    Fixtures present: {', '.join(l2_fixtures)}
    
    OCR TEXT: {ocr_str}

    RULES: 
    1. DO NOT guess random brands. Rely ONLY on the image and OCR text.
    2. Output ONLY the core brand name (e.g., 'Coca-Cola', 'Almat Oxi-Bright').
    3. You MUST return a valid JSON object exactly matching this format:
    {{
      "keywords": ["Brand 1", "Brand 2", "Brand 3"]
    }}
    """

    def _call():
        response = ollama.chat(
            model=DEFAULT_MODEL,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
            format='json',
            options={'temperature': 0.3}
        )
        try:
            result = json.loads(response['message']['content'])
            # Ensure it is a list
            kw = result.get("keywords", [])
            if not isinstance(kw, list):
                kw = [str(x) for x in kw.values()] if isinstance(kw, dict) else [str(kw)]
            return kw
        except Exception:
            return []

    return await asyncio.to_thread(_call)

def clip_l3_product_tag(image_path: str, keywords: list, clip_model, preprocess, device):
    if not keywords: return []
    import clip
    from PIL import Image

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    prompts = [f"A photo of {kw}" for kw in keywords]
    text = clip.tokenize(prompts, truncate=True).to(device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    tagged = []
    for kw, prob in zip(keywords, probs):
        tag_type = "Hard" if prob > 0.15 else "Soft" if prob > 0.05 else "Absence"
        tagged.append({
            "product": kw,
            "confidence": round(float(prob), 4),
            "tag_type": tag_type,
        })
    tagged.sort(key=lambda x: x["confidence"], reverse=True)
    return tagged

async def process_single_image(
    item: dict, metadata_map: dict, clip_model, preprocess, device: str,
    cache: dict, lock: asyncio.Lock,
) -> dict | None:
    img_path = item["image_path"]

    async with lock:
        if img_path in cache:
            return None 

    safe_print(f"\n--- L3 Tagging: {os.path.basename(img_path)} ---")
    try:
        l1_data = item.get("L1_scene") or item.get("L1", {})
        l2_data = item.get("L2_fixtures") or item.get("L2", {})
        l1_scene = l1_data.get("predicted_scene", "Unknown")
        l2_fixtures = l2_data.get("fixtures_detected", [])

        ocr_texts = extract_ocr_text(img_path)
        safe_print(f"  OCR extracted {len(ocr_texts)} text blocks")

        meta_info = metadata_map.get(img_path, {})
        store_ctx = meta_info.get("retailer_metadata", "Unknown")

        keywords = await get_dynamic_keywords(
            img_path, l1_scene, l2_fixtures, ocr_texts, store_context=store_ctx
        )

        safe_print(f"  Ollama returned {len(keywords)} keywords: {str(keywords[:5])[:120]}...")

        if not keywords:
            safe_print(f"  [SKIP] 0 keywords. Marking as empty.")
            return {
                "image_path": img_path, "L1_scene": l1_data, "L2_fixtures": l2_data,
                "L3_dynamic_keywords": [], "L3_product_tags": [], "ocr_text": ocr_texts,
                "l3_source": "empty"
            }

        l3_tags = clip_l3_product_tag(img_path, keywords, clip_model, preprocess, device)

        return {
            "image_path": img_path, "L1_scene": l1_data, "L2_fixtures": l2_data,
            "L3_dynamic_keywords": keywords, "L3_product_tags": l3_tags,
            "ocr_text": ocr_texts, "l3_source": "ollama"
        }

    except Exception as e:
        safe_print(f"  [ERROR] {os.path.basename(img_path)}: {type(e).__name__} - skipping.")
        log_error(img_path, "l3_tagging", e)
        return None

async def main():
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    with open(IN_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    # test run on 100 items
    items = items[:100]
    
    cache = load_existing_cache()
    lock = asyncio.Lock()

    tasks = [
        process_single_image(item, {}, clip_model, preprocess, device, cache, lock)
        for item in items
    ]

    results = await asyncio.gather(*tasks)
    
    for r in results:
        if r:
            cache[r["image_path"]] = r

    save_cache(cache)
    safe_print(f"Finished processing. Total cached: {len(cache)}")

if __name__ == "__main__":
    asyncio.run(main())
