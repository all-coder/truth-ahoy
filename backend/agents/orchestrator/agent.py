# necessary imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

#relative imports
from backend.agents.deep_analyst.agent import DeepAnalystAgent
from utils.helpers import get_model

model = get_model()
CoordinatorAgent = Agent(
    name="Coordinator",
    model=model,
    description="Routes all queries to DeepAnalyst for comprehensive reasoning.",
    instruction=(
        "For all user queries, transfer to DeepAnalyst using transfer_to_agent for detailed analysis and reasoning."
    ),
    sub_agents=[DeepAnalystAgent]
)