# DeepAnalystAgent.py
from google.adk.agents import BaseAgent, LlmAgent
from google.genai import types
from dotenv import load_dotenv
from typing import AsyncGenerator
from typing_extensions import override
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part
from google.adk.tools.agent_tool import AgentTool

from utils.helpers import get_model
from backend.agents.web_agent.agent import WebAgent


model = get_model()
summarizer_agent = LlmAgent(
    name="AnalystAgent",
    model=model,
    instruction=(
        """You are an information analyst. 
        Your task is to investigate the provided statement, assess credibility, 
        and produce a structured analytical report. 

        User's statement: {search_query}
        Web Analysis: {web_agent_analysis}

        Provide ONLY:
        1. **Key Findings (bullet points)** – Summarize the most important insights.  
        2. **Detailed Analysis (paragraphs)** – Cover the following aspects:  
            • Credibility assessment of the sources.  
            • Indicators of authenticity or misinformation (e.g., factual consistency, anomalies).  
            • Contextual factors: missing details, potential biases, or manipulation signals.  
            • Practical guidance for the user on how to critically evaluate similar content.  

        Your response must read like a detailed report, not a casual summary. 
        Do not add extra labels, disclaimers, or commentary beyond the required structure."""
    ),
    output_key="analysis_output",
)


class DeepAnalystAgent(BaseAgent):
    web_agent: WebAgent
    web_agent_tool: AgentTool
    deep_analyst_summarizer: LlmAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str = "DeepAnalyst"):
        """
        Initializes the DeepAnalystAgent.
        """
        web_agent = WebAgent()
        web_agent_tool = AgentTool(agent=web_agent)
        sub_agents_list = [web_agent]
        deep_analyst_summarizer = summarizer_agent
        super().__init__(
            name=name,
            description="Breaks down the user's statement, calls WebAgent to get summarized info, and provides bullet points on why the statement is right or wrong.",
            web_agent=web_agent,
            web_agent_tool=web_agent_tool,
            sub_agents=sub_agents_list,
            deep_analyst_summarizer=deep_analyst_summarizer,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the analysis workflow.
        """
        user_fact = ctx.session.state.get("search_query")
        if not user_fact:
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant", parts=[Part(text="No input provided")]
                ),
            )
            return

        ctx.session.state["search_query"] = user_fact
        async for event in self.web_agent._run_async_impl(ctx):
            pass
        web_agent_analysis = ctx.session.state.get("web_agent_analysis")
        print("--DEEP ANALYST AGENT : WEB_AGENT RESULT--")
        if not web_agent_analysis:
            state_keys = list(ctx.session.state.keys())
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant",
                    parts=[
                        Part(
                            text=f"Deep Analyst Agent : Failed Analysis, No data from WebAgent"
                        )
                    ],
                ),
            )
            return

        analysis_output = None
        async for event in self.deep_analyst_summarizer.run_async(ctx):
            analysis_output = event.content.parts[0].text
            yield event

        yield Event(
            author=self.name,
            content=Content(role="assistant", parts=[Part(text=analysis_output)]),
        )