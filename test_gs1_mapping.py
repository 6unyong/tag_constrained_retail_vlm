import asyncio
import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class GS1MappingItem(BaseModel):
    original_category: str
    gs1_family: str
    gs1_brick: str

class GS1MappingResult(BaseModel):
    mappings: list[GS1MappingItem]

async def test_mapping():
    client = genai.Client()
    
    sample_categories = [
        'Review', 'Seasonal Non-Food', 'Halloween', 'Cakes and Treats',
        'Confectionery', 'Toys', 'Soft Drinks and Mixers', 'Beauty Gifts'
    ]
    
    prompt = f"""
    You are an expert in GS1 Global Product Classification (GPC).
    I have a list of raw product categories extracted from grocery retail images.
    Please map each one to its most appropriate GS1 GPC 'Family' and 'Brick' level names.
    If a category is too abstract like 'Review', map it to something broad like 'Unclassified' or the closest guess.
    
    Categories to map:
    {json.dumps(sample_categories, ensure_ascii=False)}
    """
    
    response = await client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=GS1MappingResult,
            temperature=0.0, 
        ),
    )
    print(response.text)

asyncio.run(test_mapping())
