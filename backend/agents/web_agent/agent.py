# WebAgent.py
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.tools import FunctionTool
from utils.helpers import get_model
from utils.tools import search_through_reddit, search_using_tavily,extract_news_articles
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part
from typing import AsyncGenerator
from typing_extensions import override
from pydantic import TypeAdapter
model = get_model()

search_reddit_tool = FunctionTool(func=search_through_reddit)
search_through_news = FunctionTool(func=extract_news_articles)

reddit_agent = LlmAgent(
    name="RedditAnalyst",
    model=model,
    instruction=(
        "You are a Reddit Analyst. You must call the `search_reddit_tool` "
        "with the provided {reddit_urls} to fetch Reddit discussions. "
        "Once the tool returns results, generate a structured, in-depth report of the discussions. "
        "Your report should include:\n"
        "- A concise headline/topic\n"
        "- Key points and recurring themes in the discussions\n"
        "- Notable user opinions, arguments, or perspectives\n"
        "- Areas of consensus\n"
        "- Areas of disagreement or conflicting opinions (highlight these distinctly)\n"
        "- Any implications, trends, or takeaways from the discussion\n\n"
        "Write in a professional, analytical tone suitable for a community insights briefing. "
        "Return only the final report."
    ),
    tools=[search_reddit_tool],
    output_key="reddit_report",
)


news_agent = LlmAgent(
    name="NewsAnalyst",
    model=model,
    instruction=(
        "You are a News Analyst. You must call the `search_through_news` "
        "with the provided {news_urls} to fetch news article contents. "
        "Once the tool returns results, produce a structured, in-depth analytical report. "
        "Your report should include:\n"
        "- A clear headline/title\n"
        "- Verified key facts and main events\n"
        "- Background context and underlying causes\n"
        "- Different perspectives or stakeholder viewpoints (if available)\n"
        "- Implications and possible future developments\n"
        "- A timeline of events, if relevant\n"
        "- **Conflicting information:** If there are contradictions across sources "
        "(e.g., differing figures, causes, or interpretations), highlight them clearly "
        "in a separate section titled 'Conflicting Reports' with source distinctions.\n\n"
        "Write in a professional, analytical tone suitable for a research or intelligence briefing. "
        "Return only the final report."
    ),
    tools=[search_through_news],
    output_key="news_report",
)


query_maker = LlmAgent(
    name="QueryMaker",
    model=model,
    instruction=(
        "Generate 3 diverse search queries based on the user's statement: {search_query}. "
        "Each query should be phrased differently to capture varied aspects of the topic. "
        "At least two queries must be aimed at major news coverage (append keywords like 'news', 'latest updates', "
        "At least one query must be aimed at Reddit discussions (append 'reddit'). "
        "Return the queries strictly as a valid Python list of strings, nothing else. Example format: "
        " '['query 1', 'query 2', 'query 3']' "
    ),
    output_key="queries",
)



class WebAgent(BaseAgent):
    reddit_agent: LlmAgent
    query_maker: LlmAgent
    news_agent:LlmAgent

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str = "WebAgent"):
        super().__init__(
            name=name,
            description="Searches through the internet to gather resources and information based on the query, and present them in an unbiased and summarized format.",
            reddit_agent=reddit_agent,
            query_maker=query_maker,
            news_agent=news_agent,
        )
        print("INIT WEBAGENT")

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        user_statement = ctx.session.state.get("search_query")
        if not user_statement:
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant", parts=[Part(text="No search query provided")]
                ),
            )
            return
        #initializing the queries_list
        queries_list = None
        async for event in self.query_maker.run_async(ctx):
            if event.content and event.content.parts:
                raw_text = event.content.parts[0].text.strip()
                try:
                    type_adapter = TypeAdapter(list[str])
                    queries_list = type_adapter.validate_python(eval(raw_text))
                except Exception:
                    queries_list = [raw_text]
            yield event
        # if query_maker failed, create fallback queries
        if not queries_list or not isinstance(queries_list, list):
            print("QUERIES LIST FAILED, FAllBACK QUERIES CREATED")
            queries_list = [
                f"{user_statement} news",
                f"{user_statement} latest updates",
                f"{user_statement} reddit",
                ]

        # setting the state for queries
        ctx.session.state["queries"] = queries_list
        all_results, reddit_urls, news_urls = [], [], []
        for q in queries_list:
            tavily_results = search_using_tavily(q)
            all_results.append(tavily_results)
            
            if tavily_results.get("status") == "success":
                urls = [
                    item.get("url")
                    for item in tavily_results.get("data", [])
                    if item.get("url")
                ]
                # Reddit URLs
                reddit_urls.extend([url for url in urls if "reddit.com" in url])
                # News URLs (everything else)
                news_urls.extend([url for url in urls if "reddit.com" not in url])

        # loading the urls onto state
        ctx.session.state["reddit_urls"] = reddit_urls
        ctx.session.state["news_urls"] = news_urls

        # we are doing reddit and news analysis alone
        reddit_analysis = None
        if reddit_urls:
            reddit_urls_to_use = reddit_urls[:2]  
            async for event in self.reddit_agent.run_async(ctx):
                reddit_analysis = event.content.parts[0].text
                yield event

        news_analysis = None
        if news_urls:
            news_urls_to_use = news_urls[:4] 
            async for event in self.news_agent.run_async(ctx):
                news_analysis = event.content.parts[0].text
                yield event
        

        ctx.session.state["news_analysis"] = news_analysis
        ctx.session.state["reddit_analysis"] = reddit_analysis

        # combined analysis within one set of strings
        web_agent_analysis = (
            f"News Analysis:\n{news_analysis}\n\n"
            f"Reddit Analysis:\n{reddit_analysis}"
        )

        print(web_agent_analysis)

        ctx.session.state["web_agent_analysis"] = web_agent_analysis
        result = {
            "status": "success",
            "queries": queries_list,
            "search_results": all_results,
            "web_agent_analysis": web_agent_analysis,
        }

        ctx.session.state["web_agent_result"] = result
        yield Event(
            author=self.name,
            content=Content(
                role="assistant",
                parts=[
                    Part(
                        text=f"Web Agent Analysis Completed"
                    )
                ],
            ),
        )
        
