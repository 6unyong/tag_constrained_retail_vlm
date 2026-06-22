import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Load Environment Variables
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

client = genai.Client()

class KanopsMappingItem(BaseModel):
    original_category: str = Field(..., description="The original raw category extracted from the image")
    kanops_parent: str = Field(..., description="The Kanops Master Taxonomy parent category (e.g., 'Ambient Grocery', 'Seasonal')")
    kanops_subcategory: str = Field(..., description="The Kanops subcategory (e.g., 'Canned Fish', 'Christmas Decorations')")

class KanopsMappingResult(BaseModel):
    mappings: List[KanopsMappingItem]

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
)
async def map_categories_to_kanops(categories: List[str]) -> KanopsMappingResult:
    # We load Kanops_Taxonomy.md to provide it to the prompt
    with open("Kanops_Taxonomy.md", "r", encoding="utf-8") as f:
        taxonomy_text = f.read()

    prompt = f"""
    You are an expert in Retail Category Classification.
    I have a list of raw product categories extracted from grocery retail images.
    Please map each one to its most appropriate parent category and subcategory based strictly on the provided Kanops Classification Taxonomy.
    
    Kanops Taxonomy Reference:
    {taxonomy_text}
    
    Categories to map:
    {json.dumps(categories, ensure_ascii=False)}
    
    Return strict JSON matching the schema. If a category cannot be confidently placed, use 'Review' for both parent and subcategory.
    """
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=KanopsMappingResult,
            temperature=0.0, 
        ),
    )
    
    if not response.text:
         raise Exception("Received empty response from Gemini.")
         
    return KanopsMappingResult.model_validate_json(response.text)

async def run_kanops_mapping():
    input_path = "data/cache/corpus_induction_results.json"
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Extract unique categories
    unique_categories = set()
    for item in results:
        if "error" in item:
            continue
        if "product_categories" in item:
            for cat in item["product_categories"]:
                unique_categories.add(cat)
                
    unique_categories = list(unique_categories)
    print(f"Found {len(unique_categories)} unique product categories to map: {unique_categories}")
    
    if not unique_categories:
        print("No valid categories to map extracted from the previous step.")
        return

    print("Sending to Gemini for Kanops Taxonomy Mapping...")
    try:
        mapping_result = await map_categories_to_kanops(unique_categories)
        
        out_path = "data/cache/kanops_mappings.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mapping_result.model_dump(), f, indent=4, ensure_ascii=False)
            
        print(f"\nKanops Mapping complete! Results saved to {out_path}")
        print("Preview of mappings:")
        print(json.dumps(mapping_result.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Failed to map categories: {e}")

if __name__ == "__main__":
    asyncio.run(run_kanops_mapping())
