from google.adk.agents import LlmAgent
from utils.helpers import get_model

model = get_model()

SummaryAgent = LlmAgent(
    name="SummarizerAgent",
    model=model,
    instruction=(
        """You are an information summarizer.
       User's statement: {search_query}
       Source summary: {reddit_summary}

       Provide ONLY:
       - A concise summary of the information in bullet points.
       - A clear explanation in detail of why the user's statement is correct or incorrect.

       Do NOT include commentary, opinions, or extra text beyond the summary and explanation."""
    ),
    output_key="analysis_output",
)
