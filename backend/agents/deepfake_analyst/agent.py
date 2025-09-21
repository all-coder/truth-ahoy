# DeepFakeAnalyst.py
from google.adk.agents import BaseAgent
from typing import AsyncGenerator
from typing_extensions import override
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part
from google.adk.tools import FunctionTool
from utils.vision import deep_image_analysis

# cloud vision tools for image analysis
# reverse_image_search_sources_tool = FunctionTool(func=reverse_image_search_sources)
# detect_landmarks_present_in_image_tool = FunctionTool(func=detect_landmarks_present_in_image)

class DeepFakeAnalyst(BaseAgent):
    def __init__(self, name: str = "DeepFakeAnalyst"):
        super().__init__(name=name)

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        images = ctx.session.state.get("image_urls", [])
        print("IMAGES URL FROM AGENT",images)
        if len(images)==0:
            ctx.session.state["image_analysis"] = "No image analysis was done"
            yield Event(
                author=self.name,
                content=Content(role="assistant", parts=[Part(text="No Deepfake image analysis was done")])
            )
            return
        print("DOING IMAGE ANALYSIS")
        # only one image for now
        deep_image_result = deep_image_analysis(images[0])
        ctx.session.state["image_analysis"] = deep_image_result

        yield Event(
            author=self.name,
            content=Content(role="assistant", parts=[Part(text="Deepfake and deep image analysis is done")])
        )

