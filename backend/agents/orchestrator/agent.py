# necessary imports
import os
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
load_dotenv()

# relative imports
from backend.agents.deep_analyst.agent import DeepAnalystAgent

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

if USE_OLLAMA:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    model = LiteLlm(
        api_base='http://localhost:11434/v1',
        model=f'openai/{OLLAMA_MODEL}',
        api_key='dummy'
    )
else:
    model = os.getenv("GEMINI_MODEL")

CoordinatorAgent = Agent(
    name="Coordinator",
    model=model,
    description="Routes all queries to DeepAnalyst for comprehensive reasoning.",
    instruction=(
        "For all user queries, transfer to DeepAnalyst using transfer_to_agent for detailed analysis and reasoning."
    ),
    sub_agents=[DeepAnalystAgent]
)