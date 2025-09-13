# WebAgent.py
from google.adk.tools.agent_tool import AgentTool
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.tools import FunctionTool
from utils.helpers import get_model
from utils.tools import search_through_reddit, search_using_tavily
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part
from typing import AsyncGenerator
from typing_extensions import override

model = get_model()

search_reddit_tool = FunctionTool(func=search_through_reddit)
reddit_agent = LlmAgent(
    name="RedditAgent",
    model=model,
    instruction=(
        "You have been provided with Reddit URLs. "
        "Use the `search_through_reddit` tool to scrape content from these URLs and summarize it. "
        "Provide a concise, well-structured summary of the Reddit discussions."
    ),
    tools=[search_reddit_tool],
    output_key="reddit_summary"
)

query_maker = LlmAgent(
    name="QueryMaker",
    model=model,
    instruction="Generate two diverse search queries based on the user's statement: {search_query}. Append the word 'reddit' at the end of each query. Return as a simple list.",
    output_key="queries"
)

reddit_agent_tool = AgentTool(agent=reddit_agent)
query_maker_tool = AgentTool(agent=query_maker)

class WebAgent(BaseAgent):
    reddit_agent_tool: AgentTool
    query_maker_tool: AgentTool
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "WebAgent"):
        super().__init__(
            name=name,
            description="Searches through the internet to gather resources and information based on the query, and present them in an unbiased and summarized format.",
            reddit_agent_tool=reddit_agent_tool,
            query_maker_tool=query_maker_tool,
        )
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        user_statement = ctx.session.state.get("search_query")
        if not user_statement:
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant",
                    parts=[Part(text="No search query provided")]
                )
            )
            return
        async for event in self.query_maker_tool.agent.run_async(ctx):
            pass 

        queries_list = ctx.session.state.get("queries", [])
        if not queries_list:
            queries_list = [f"{user_statement} reddit", f"{user_statement} discussion reddit"]
            print(f"Fallback queries: {queries_list}")

        all_results = []
        reddit_urls = []

        for q in queries_list:
            tavily_results = search_using_tavily(q)
            all_results.append(tavily_results)
            if tavily_results.get("status") == "success":
                urls = [item.get("url") for item in tavily_results.get("data", []) if item.get("url")]
                reddit_urls.extend([url for url in urls if "reddit.com" in url])

        reddit_summary = None
        if reddit_urls:
            ctx.session.state["urls"] = reddit_urls
            urls_message = f"Analyze these Reddit URLs: {', '.join(reddit_urls)}"
            temp_ctx = ctx.copy()
            temp_ctx.session.state["current_message"] = urls_message
            async for event in self.reddit_agent_tool.agent.run_async(temp_ctx):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        reddit_summary = event.content.parts[0].text
                        ctx.session.state["reddit_summary"] = reddit_summary

        # Store the final result
        result = {
            "status": "success",
            "queries": queries_list,
            "search_results": all_results,
            "reddit_summary": reddit_summary
        }
        
        ctx.session.state["web_agent_result"] = result

        # Yield a completion event
        yield Event(
            author=self.name,
            content=Content(
                role="assistant",
                parts=[Part(text=f"Web search completed. Found {len(reddit_urls)} Reddit URLs. Summary generated.")]
            )
        )
