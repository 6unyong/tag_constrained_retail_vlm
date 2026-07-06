import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

def main():
    json_path = "data/eval_results/evaluation_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize Gemini client
    client = genai.Client()
    
    print(f"Translating {len(data)} items to Korean...")
    
    for i, item in enumerate(data):
        if item.get("translated"):
            continue
            
        print(f"Translating item {i+1}/{len(data)}...")
        prompt = f"""
Translate the following two English image captions into Korean. Maintain the exact factual details, tone, and any brand names (you can keep brands in English or transliterate them). Do not fix errors in the original text; if the original text is weird or hallucinates, keep the translation weird or hallucinated.

Caption A:
{item['baseline_caption']}

Caption B:
{item['mop_caption']}

Output exactly in this JSON format:
{{
  "caption_a_kr": "...",
  "caption_b_kr": "..."
}}
"""
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            
            res_dict = json.loads(response.text)
            item['baseline_caption'] = res_dict['caption_a_kr']
            item['mop_caption'] = res_dict['caption_b_kr']
            item['translated'] = True
            
            # Save incrementally
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            time.sleep(0.5) # rate limit prevention
            
        except Exception as e:
            print(f"Error translating item {i+1}: {e}")
            
    print("Translation complete!")

if __name__ == "__main__":
    main()
