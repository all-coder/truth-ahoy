# necessary imports
from google.adk.agents import LlmAgent
from google.genai import types
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

# relative imports
from utils.helpers import get_model

model = get_model()

FactCheckerAgent = LlmAgent(
    name="FactChecker",
    model=model,
    instruction="""Think step by step internally to verify the claim based on your knowledge.
    If confident, output JSON: {"status":"resolved","answer":"..."}
    If not confident, output JSON: {"status":"needs_analysis","reason":"..."}
    Do not reveal internal reasoning.""",
    input_schema=None,
    output_key="fact_checker_output"
)
