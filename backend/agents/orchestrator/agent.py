from google.adk.agents import BaseAgent
from agents.deep_analyst.agent import DeepAnalystAgent
from utils.helpers import get_model
from google.genai.types import Content, Part
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator
from typing_extensions import override

model = get_model()

class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that receives user facts/statements and delegates analysis to DeepAnalyst.
    """
    deep_analyst: DeepAnalystAgent
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "Orchestrator"):
        """
        Initializes the OrchestratorAgent.
        """
        deep_analyst_agent = DeepAnalystAgent()
        sub_agents_list = [deep_analyst_agent]
        super().__init__(
            name=name,
            description="Receives the user's fact or statement, stores it, and delegates analysis to DeepAnalyst.",
            deep_analyst=deep_analyst_agent,
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Implements the orchestration logic for fact analysis workflow.
        """
        if ctx.user_content and ctx.user_content.parts:
            user_fact = ctx.user_content.parts[0].text
        else:
            yield Event(
                author="system",
                content=Content(
                    role="system",
                    parts=[Part(text="No input provided")]
                )
            )
            return
        ctx.session.state["search_query"] = user_fact
        async for event in self.deep_analyst.run_async(ctx):
            yield event