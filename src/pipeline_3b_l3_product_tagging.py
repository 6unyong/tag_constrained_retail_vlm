"""
Task 10: L3 Product Tagging
Step 1 — Send L1/L2 context + GS1 mappings to Gemini Flash → get dynamic product keyword list
Step 2 — Use those keywords as CLIP zero-shot prompts → produce L3 probabilistic tags

10K Resilience features:
- Dict-based checkpoint: already-tagged images are skipped on re-run (no double Gemini billing)
- Per-image try/except: one bad image won't kill the entire run
- Auto-save every 50 images to prevent data loss on crash/OOM
"""
import os
import json
import asyncio
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

import torch
from paddleocr import PaddleOCR

load_dotenv()

from google import genai
from google.genai import types
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

client = genai.Client()
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

# ── Constants ──
IN_PATH = "data/cache/l1_l2_tag_results.json"
OUT_PATH = "data/cache/l1_l2_l3_tag_results.json"
ERROR_LOG = "data/cache/error_log.txt"
AUTOSAVE_INTERVAL = 50

# ── Pydantic schema for Gemini response ──
class DynamicProductKeywords(BaseModel):
    keywords: List[str] = Field(
        ..., description="List of specific product names likely visible in this retail scene"
    )

def log_error(img_path: str, stage: str, error: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[pipeline_3b|{stage}] {img_path} | {type(error).__name__}: {error}\n")

def load_existing_cache() -> dict:
    """
    Load previously saved L3 results as {image_path: result} for checkpoint resumption.
    This is the primary guard against Gemini double-billing.
    """
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
    """Extract visible text from image using PaddleOCR."""
    result = ocr.ocr(image_path, cls=True)
    texts = []
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:
                texts.append({"text": text, "confidence": round(float(confidence), 3)})
    return texts

@retry(
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
)
async def get_dynamic_keywords(
    l1_scene: str, l2_fixtures: list, gs1_categories: list,
    ocr_texts: list, store_context: str = "Unknown UK Supermarket"
) -> list:
    """
    Call Gemini Flash to generate context-aware product keywords.
    Wrapped with tenacity retry to handle 429 Rate Limit errors gracefully.
    """
    ocr_str = ", ".join([t['text'] for t in ocr_texts]) if ocr_texts else "No readable text"

    prompt = f"""
    You are a grocery retail merchandising expert for '{store_context}'.
    Given the following store context and VISUALLY EXTRACTED TEXT (OCR), generate a list of 10-20 specific product names 
    that would likely be visible on these fixtures.

    Scene zone: {l1_scene}
    Fixtures present: {', '.join(l2_fixtures)}
    Known product categories (GS1 GPC): {', '.join(gs1_categories)}
    
    VISUAL TEXT EXTRACTED FROM IMAGE (OCR): {ocr_str}

    CRITICAL RULES: 
    1. DO NOT guess random brands just because of the store_context. Strictly rely on the VISUAL TEXT EXTRACTED to infer specific products and brand names.
    2. DO NOT include packaging sizes, volumes (e.g., 500ml, 2L, multipack, gm). Output ONLY the core base product brand name (e.g., 'Coca-Cola Original Taste', 'Coca-Cola Zero Sugar').
    3. Return specific, visually identifiable product names matching the actual texts (Do NOT mix competitor brands).
    Return strict JSON matching the schema.
    """
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DynamicProductKeywords,
            temperature=0.3,
        ),
    )
    if not response.text:
        raise Exception("Empty response from Gemini")
    result = DynamicProductKeywords.model_validate_json(response.text)
    return result.keywords

def clip_l3_product_tag(image_path: str, keywords: list, clip_model, preprocess, device):
    """Run CLIP zero-shot against dynamic keywords to produce L3 tags."""
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

async def run_l3_tagging():
    with open(IN_PATH, "r") as f:
        l1l2_results = json.load(f)
    with open("data/cache/gs1_mappings.json", "r") as f:
        gs1_data = json.load(f)

    try:
        with open("data/cache/metadata_mapped.json", "r") as f:
            metadata_map = json.load(f)
    except FileNotFoundError:
        metadata_map = {}

    gs1_categories = [m["gs1_brick"] for m in gs1_data["mappings"]]

    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # --- Checkpoint: skip images already in the output cache ---
    # This is the primary double-billing guard for Gemini API.
    cache = load_existing_cache()
    already_done = len(cache)
    if already_done > 0:
        print(f"[CHECKPOINT] {already_done} images already have L3 tags. Skipping (no Gemini re-call).")

    new_count = 0
    for item in l1l2_results:
        img_path = item["image_path"]

        if img_path in cache:
            continue  # Already processed — skip Gemini call entirely

        print(f"\n--- L3 Tagging: {os.path.basename(img_path)} ---")

        try:
            l1_scene = item["L1"]["predicted_scene"]
            l2_fixtures = item["L2"]["fixtures_detected"]

            ocr_texts = extract_ocr_text(img_path)
            print(f"  OCR extracted {len(ocr_texts)} text blocks")

            meta_info = metadata_map.get(img_path, {})
            store_ctx = meta_info.get("retailer_metadata", "Unknown")

            # Gemini API call — wrapped with tenacity for 429 handling
            keywords = await get_dynamic_keywords(
                l1_scene, l2_fixtures, gs1_categories, ocr_texts, store_context=store_ctx
            )
            print(f"  Gemini returned {len(keywords)} keywords: {keywords[:5]}...")

            l3_tags = clip_l3_product_tag(img_path, keywords, clip_model, preprocess, device)

            cache[img_path] = {
                "image_path": img_path,
                "L1": item["L1"],
                "L2": item["L2"],
                "L3_dynamic_keywords": keywords,
                "L3_product_tags": l3_tags,
                "ocr_text": ocr_texts
            }
            new_count += 1

            # Auto-save every AUTOSAVE_INTERVAL images
            if new_count % AUTOSAVE_INTERVAL == 0:
                save_cache(cache)
                print(f"  [AUTO-SAVE] Checkpoint at {new_count} new images (total cache: {len(cache)}).")

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: {e} — skipping.")
            log_error(img_path, "l3_tagging", e)
            continue

    # Final save
    save_cache(cache)
    print(f"\nSaved combined L1+L2+L3 results to {OUT_PATH}")
    print(f"Processed this session: {new_count} | Total in cache: {len(cache)}")

if __name__ == "__main__":
    asyncio.run(run_l3_tagging())
