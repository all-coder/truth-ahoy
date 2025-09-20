import os
from google.adk.models.lite_llm import LiteLlm
import requests
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv

load_dotenv()

# gets the model
def get_model():
    USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
    if USE_OLLAMA:
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        model = LiteLlm(
            api_base='http://localhost:11434/v1',
            model=f'openai/{OLLAMA_MODEL}',
            api_key='dummy'
        )
    else:
        # for some reason, gemini-2.5-flash gets loaded despite changing in the env, so mandatorily setting up gemini-1.5-flash
        model = "gemini-1.5-flash"
    return model

def get_service_key():
    path = os.getenv("GOOGLE_CLOUD_SERVICE_KEY")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Service account key not found at: {path}")
    
    return path

def get_table_id():
    if(os.getenv("TABLE_ID")):
        return os.getenv("TABLE_ID")
    raise Exception("Table ID not found")

# returns a PIL.Image() object, and base64 encoded string
def fetch_image_from_url(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
        "Sec-CH-UA": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        img_bytes = response.content
        img = Image.open(BytesIO(img_bytes))
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        return img, img_b64
    except:
        return -1
