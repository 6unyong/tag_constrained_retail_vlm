import os
import asyncio
import json
from glob import glob
from pydantic import BaseModel, Field
from typing import List

from src.utils.async_gemini import generate_structured_vision_async

# 1. Define the Discovery Schema for Ontology (Corpus Induction Phase)
class CorpusInductionResult(BaseModel):
    image_file: str = Field(..., description="Name of the processed image file")
    scene_description: str = Field(..., description="Overall location context, e.g., ambient aisle, endcap, checkout")
    fixtures_found: List[str] = Field(..., description="List of physical structures holding products")
    product_categories: List[str] = Field(..., description="General grocery categories found, e.g., soft drinks, fresh produce")
    operational_issues: List[str] = Field(..., description="Any visible issues like empty shelves, messy displays, or promotional flags")

async def process_image_for_induction(image_path: str) -> dict:
    prompt = """
    You are an expert in Retail Visual Merchandising and Grocery Store Operations.
    Please analyze this image and extract raw descriptive noun phrases (Corpus Induction).
    Focus on:
    1. The general scene/zone (e.g. ambient aisle, chiller, checkout).
    2. The physical fixtures present (e.g. gondola shelving, cardboard dump bin, promotional endcap).
    3. The product categories visible (broad GS1 GPC level, like 'carbonated beverages', 'cookies').
    4. Any operational states or issues (e.g. fully stocked, out of stock gap, messy, active promotion).
    Return strict JSON matching the schema.
    """
    print(f"Processing {image_path}...")
    try:
        result = await generate_structured_vision_async(
            image_path=image_path,
            prompt=prompt,
            response_schema=CorpusInductionResult,
            model_name="gemini-2.5-pro"
        )
        # Ensure the filename is injected
        result.image_file = os.path.basename(image_path)
        return result.model_dump()
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        return {"image_file": os.path.basename(image_path), "error": str(e)}

async def run_corpus_induction():
    image_dir = "data/processed"
    images = glob(os.path.join(image_dir, "*.jpg"))
    
    if not images:
        print(f"No images found in {image_dir}.")
        return

    # In a full run, we would limit this to a stratified sample of 100 images
    print(f"Found {len(images)} images for Corpus Induction Testing.")
    
    # Process concurrently using asyncio.gather
    tasks = [process_image_for_induction(img) for img in images]
    results = await asyncio.gather(*tasks)
    
    # Save the induction corpus
    os.makedirs("data/cache", exist_ok=True)
    out_path = "data/cache/corpus_induction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nCorpus Induction complete! Results saved to {out_path}")
    print("Preview of first result:")
    if results:
        print(json.dumps(results[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(run_corpus_induction())
