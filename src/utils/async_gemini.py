import os
import asyncio
from typing import Type, TypeVar
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Load Environment Variables
load_dotenv()

from google import genai
from google.genai import types

# Initialize the async client
client = genai.Client()

T = TypeVar("T", bound=BaseModel)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
)
async def generate_structured_vision_async(
    image_path: str,
    prompt: str,
    response_schema: Type[T],
    model_name: str = "gemini-2.5-pro"
) -> T:
    """
    Calls Gemini API with an image and a prompt, returning a tightly controlled Pydantic structure.
    Implements exponential backoff to handle rate limits.
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image at {image_path}: {e}")

    try:
        # Async generation
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2, # Low temperature for more deterministic extraction
            ),
        )
        
        if not response.text:
             raise Exception("Received empty response from Gemini.")
             
        return response_schema.model_validate_json(response.text)
    
    except Exception as e:
        print(f"Error calling Gemini API for {image_path}: {e}")
        raise e
