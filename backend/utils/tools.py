# necessary imports
import os
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import List, Dict, Any

# relative imports
from utils.bots.reddit import RedditRetriever

# please do ensure that each tool (if going to be used for agent or not) definition includes properly annotated docstrings,
# as they are essential for agents to correctly utilize the tools.

load_dotenv()


def search_using_tavily(query: str) -> Dict[Any]:
    """
    Retrieves search results from the Tavily API.

    Args:
        query (str): The search query which will be used to return a list of relevant and ranked urls.

    Returns:
        dict: A dictionary containing the search results.
        Includes a status key indicating success or failure.
        If successful, the data key will contain the search results.
        Otherwise will contain an error message.
    """
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(query, max_results=10)
    except Exception as e:
        return {"status": "failure", "error_message": str(e)}
    return {"status": "success", "data": response["results"]}


def search_through_reddit(urls: List[str]) -> Dict[Any]:
    """
    Searches for the given URLs on Reddit.
    Args:
        urls (List[str]): A list of Reddit URLs to search for.

    Returns:
        dict: A dictionary containing the search results.
        Includes a status key indicating success or failure.

    """
    if len(urls) == 0:
        return {"status": "failure", "error_message": "No Reddit URLs provided"}
    try:
        reddit = RedditRetriever()
        results = reddit.get_and_process_data(urls, max_depth=3)
    except Exception as e:
        return {"status": "failure", "error_message": str(e)}
    return {"status": "success", "data": results}
