# necessary imports
import os
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import List, Dict, Any

# relative imports
from utils.bots.reddit import RedditRetriever
from utils.vision import reverse_image_search, detect_landmarks

# please do ensure that each tool (if going to be used for agent or not) definition includes properly annotated docstrings,
# as they are essential for agents to correctly utilize the tools.

load_dotenv()


def search_using_tavily(query: str) -> Dict[str, Any]:
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


def search_through_reddit(urls: List[str]) -> Dict[str, Any]:
    """
    Searches for the given URLs on Reddit.

    Args:
        urls (List[str]): A list of urls to search for on Reddit.

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
        return {"status": "Reddit Search Failed", "error_message": str(e)}
    return {"status": "success", "data": results}


def reverse_image_search_sources(image_base64: str) -> Dict[str, Any]:
    """
    Performs a reverse image search on the given base64-encoded image.

    Args:
        image_base64 (str): Base64-encoded string of the image, with or without
                            the 'data:image/...;base64,' prefix.

    Returns:
        dict: A dictionary containing the reverse image search results.
        Includes a status key indicating success or failure.
    """
    if not image_base64:
        return {"status": "failure", "error_message": "No image provided"}
    try:
        results = reverse_image_search(image_base64)
        if "error" in results:
            return {"status": "failure", "error_message": results["error"]}
    except Exception as e:
        return {"status": "failure", "error_message": str(e)}
    return {"status": "success", "data": results}


def detect_landmarks_present_in_image(image_base64: str) -> Dict[str, Any]:
    """
    Detects landmarks in the given base64-encoded image and returns their
    names along with geographic coordinates (latitude and longitude).

    Args:
        image_base64 (str): Base64-encoded string of the image, with or without
                            the 'data:image/...;base64,' prefix.

    Returns:
        dict: A dictionary containing the detected landmarks and their locations.
        Includes a status key indicating success or failure.
    """
    if not image_base64:
        return {"status": "failure", "error_message": "No image provided"}
    try:
        results = detect_landmarks(image_base64)
        if "error" in results:
            return {"status": "failure", "error_message": results["error"]}
    except Exception as e:
        return {"status": "failure", "error_message": str(e)}
    return {"status": "success", "data": results}
