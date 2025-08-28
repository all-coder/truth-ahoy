# necessary imports
from google.adk.agents import LlmAgent
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

# relative imports
from utils.helpers import get_model
model = get_model()

DeepAnalystAgent = LlmAgent(
    name="DeepAnalyst",
    model=model,
    description="Handles complex multi-step analysis with structured outputs.",
    instruction=(
        "Break down complex questions into logical steps, identify risks, assumptions, "
        "and provide a thorough conclusion. Use tools when necessary to ground your answer."
    ),
    output_key="analysis_output",
    generate_content_config=types.GenerateContentConfig(temperature=0.7)
)
