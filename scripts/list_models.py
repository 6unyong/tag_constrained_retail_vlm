from google import genai
from dotenv import load_dotenv
load_dotenv()
client = genai.Client()
for m in client.models.list():
    if "generateContent" in str(getattr(m, "supported_actions", [])):
        print(m.name)
