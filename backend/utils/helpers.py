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