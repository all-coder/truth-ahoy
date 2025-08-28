from google.adk.tools.agent_tool import AgentTool
from google.adk.agents import BaseAgent, Agent
from utils.helpers import get_model
from utils.tools import search_through_reddit, search_using_tavily

model = get_model()

reddit_agent = Agent(
    name="Reddit Agent",
    model=model,
    instruction=(
        "First, use the `search_through_reddit` tool to scrape the content "
        "from the given URLs. Then read the returned data and summarize it into a "
        "concise, well-structured response for the user."
    ),
    tools=[search_through_reddit],
)

reddit_agent_tool = AgentTool(agent=reddit_agent)

class WebAgent(BaseAgent):
    name: str = "WebAgent"
    description: str = (
        "Searches through the internet to gather resources and information based on the query, and present them in an unbiased and summarized format."
    )

    async def _run_async_impl(self, ctx):
        search_query = ctx.session.state.get("search_query")
        if not search_query:
            return {"status": "failure", "error_message": "No search query provided"}

        tavily_results = await ctx.run_tool(search_using_tavily, {"query": search_query})
        urls = [item.get("url") for item in tavily_results.get("results", []) if item.get("url")]

        reddit_urls = [url for url in urls if "reddit.com" in url]

        reddit_summary = None
        if reddit_urls:
            reddit_summary = await ctx.run_tool(reddit_agent_tool, {"urls": reddit_urls})

        return {
            "status": "success",
            "search_results": tavily_results,
            "reddit_summary": reddit_summary
        }
