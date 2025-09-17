import os
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm


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
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return model

def get_service_key():
    path = os.getenv("GOOGLE_CLOUD_SERVICE_KEY")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Service account key not found at: {path}")
    return path
