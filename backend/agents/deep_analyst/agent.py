import os
from google.adk.agents import Agent
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

if USE_OLLAMA:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")   
    model = LiteLlm(
        api_base='http://localhost:11434/v1',
        model=f'openai/{OLLAMA_MODEL}',
        api_key='dummy'
    )
else:
    model = "gemini-1.5-flash"

DeepAnalystAgent = Agent(
    name="DeepAnalyst",
    model=model,
    description="Handles complex multi-step analysis with structured outputs.",
    instruction=(
        "Break down complex questions into logical steps, identify risks, assumptions, "
        "and provide a thorough conclusion. Use tools when necessary to ground your answer."
    ),
    generate_content_config=types.GenerateContentConfig(temperature=0.7)
)