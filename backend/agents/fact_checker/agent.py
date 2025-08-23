import os
from google.adk.agents import Agent
from google.genai import types
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import agent_tool
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

FactCheckerAgent = Agent(
    name="FactChecker",
    model=model,
    description="Performs quick factual verification; if unclear or complex, signals for deeper analysis.",
 instruction=(
        "Think step by step internally to verify the claim based on your knowledge. "
        "If confident, output JSON: {\"status\":\"resolved\",\"answer\":\"...\"} "
        "If not confident, output JSON: {\"status\":\"needs_analysis\",\"reason\":\"...\"} "
        "Do not reveal internal reasoning."
    ),
    generate_content_config=types.GenerateContentConfig(temperature=0)
)


FactCheckerTool = agent_tool.AgentTool(agent=FactCheckerAgent)


