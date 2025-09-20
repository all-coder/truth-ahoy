# DeepFakeAnalyst.py
from google.adk.agents import BaseAgent
from typing import AsyncGenerator
from typing_extensions import override
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part
from google.adk.tools import FunctionTool
from utils.tools import reverse_image_search_sources, detect_landmarks_present_in_image

# cloud vision tools for image analysis
reverse_image_search_sources_tool = FunctionTool(func=reverse_image_search_sources)
detect_landmarks_present_in_image_tool = FunctionTool(func=detect_landmarks_present_in_image)

class DeepFakeAnalyst(BaseAgent):
    def __init__(self, name: str = "DeepFakeAnalyst"):
        super().__init__(name=name)

    @override
    async def _run_async_impl( 
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        analysis_output = None
        
        yield Event(
            author=self.name,
            content=Content(role="assistant", parts=[Part(text=analysis_output)]),
        )
