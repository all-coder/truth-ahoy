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
from backend.agents.deepfake_analyst.agent import DeepFakeAnalyst


model = get_model()
summarizer_agent = LlmAgent(
    name="AnalystAgent",
    model=model,
    instruction=(
        """You are an information analyst. Your task is to carefully dissect the given statement and determine whether it might be fake, misleading, or accurate. Be detailed and precise—no vague summaries or generic reports. Explain clearly **why** the claim might be true or false, with concrete evidence.  

    User's statement: {search_query}  
    Web findings: {web_agent_analysis}  
    Image findings: {image_analysis}  

    Provide ONLY the following sections:  

    1. **Key Observations (bullet points)** – Break down the main points from the statement, web findings, and image findings. Include:  
    • Contradictions, inconsistencies, or unsupported claims.  
    • Verified facts or historical context.  
    • Any suspicious patterns or manipulations detected in the images.  

    2. **Detailed Reasoning (paragraph)** – Explain in detail why the claim might be true, false, or misleading. Cover:  
    • Evidence supporting or contradicting the claim.  
    • Specific points from web sources and image analysis that raise doubt or confirm it.  
    • Logical gaps, exaggerations, or missing context that affect credibility.  

    3. **Image Insights (if applicable, bullet points)** – Highlight major findings from image analysis:  
    • Signs of manipulation, deepfake artifacts, or inconsistencies.  
    • Alignment or conflict with textual claims.  

    4. **Final Assessment (short statement)** – Conclude clearly: “Likely Fake”, “Misleading”, or “Genuine”.  

    Output ONLY the analysis in the requested format. Do NOT include headers, labels, or any text like ‘AnalystAgent Report’. Focus solely on clear, evidence-backed evaluation of the claim."""
        ),
        output_key="analysis_output",
)




class DeepAnalystAgent(BaseAgent):
    web_agent: WebAgent
    deepfake_agent : DeepFakeAnalyst
    web_agent_tool: AgentTool
    deep_analyst_summarizer: LlmAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str = "DeepAnalyst"):
        """
        Initializes the DeepAnalystAgent.
        """
        web_agent = WebAgent()
        deepfake_agent = DeepFakeAnalyst()
        web_agent_tool = AgentTool(agent=web_agent)
        sub_agents_list = [web_agent,deepfake_agent]
        deep_analyst_summarizer = summarizer_agent
        super().__init__(
            name=name,
            description="Breaks down the user's statement, calls WebAgent to get summarized info, and provides bullet points on why the statement is right or wrong.",
            web_agent=web_agent,
            deepfake_agent = deepfake_agent,
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
        
        images = ctx.session.state["image_urls"]
        print("IMAGES_URLS",images)
        if images:
            async for event in self.deepfake_agent.run_async(ctx):
                yield event
            image_analysis = ctx.session.state.get("image_analysis")
            if not image_analysis:
                ctx.session.state["image_analysis"] = "Image analysis completed but no data returned"
        else:
            ctx.session.state["image_analysis"] = "No image analysis was done"
    
        analysis_output = None
        async for event in self.deep_analyst_summarizer.run_async(ctx):
            analysis_output = event.content.parts[0].text
            yield event

        yield Event(
            author=self.name,
            content=Content(role="assistant", parts=[Part(text=analysis_output)]),
        )