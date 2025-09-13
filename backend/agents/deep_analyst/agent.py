# DeepAnalystAgent.py
from google.adk.agents import BaseAgent
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

load_dotenv()
model = get_model()

class DeepAnalystAgent(BaseAgent):
    web_agent: WebAgent
    web_agent_tool: AgentTool
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "DeepAnalyst"):
        """
        Initializes the DeepAnalystAgent.
        """
        web_agent = WebAgent()
        web_agent_tool = AgentTool(agent=web_agent)
        sub_agents_list = [web_agent]
        super().__init__(
            name=name,
            description="Breaks down the user's statement, calls WebAgent to get summarized info, and provides bullet points on why the statement is right or wrong.",
            web_agent=web_agent,
            web_agent_tool=web_agent_tool,
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Implements the analysis workflow.
        """
        user_fact = ctx.session.state.get("search_query")
        if not user_fact:
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant",
                    parts=[Part(text="No input provided")]
                )
            )
            return

        ctx.session.state["search_query"] = user_fact
        async for event in self.web_agent._run_async_impl(ctx):
            pass
        web_result = ctx.session.state.get("web_agent_result")  # Assuming WebAgent saves result here
        reddit_summary = None
        if web_result:
            reddit_summary = web_result.get("reddit_summary")

        if not reddit_summary:
            state_keys = list(ctx.session.state.keys())
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant", 
                    parts=[Part(text=f"No data from WebAgent. Available state keys: {state_keys}. State: {dict(ctx.session.state)}")]
                )
            )
            return

        prompt = (
            f"User's statement: {user_fact}\n"
            f"Web summary: {reddit_summary}\n\n"
            "Provide clear bullet points explaining whether the user's statement is correct or incorrect, and why."
        )

        response = await ctx.model.generate_content(
            [prompt],
            config=types.GenerateContentConfig(temperature=0.5)
        )

        analysis_output = ""
        if response.candidates:
            analysis_output = response.candidates[0].content.parts[0].text
        
        ctx.session.state["analysis_output"] = analysis_output
        yield Event(
            author=self.name,
            content=Content(
                role="assistant",
                parts=[Part(text=analysis_output)]
            )
        )