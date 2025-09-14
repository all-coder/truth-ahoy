from typing import List, Dict, Any
from trafilatura import fetch_url, extract
from urllib.parse import urlparse

NEWS_DOMAINS = [
    "hindustantimes.com",
    "indiatoday.in",
    "timesofindia.indiatimes.com",
    "bbc.com",
    "news.sky.com",
    "aljazeera.com"
]

def extract_news(urls: List[str]) -> Dict[str, Any]:
    result = {}
    for i in urls:
        try:
            domain = urlparse(i).netloc
            if not any(d in domain for d in NEWS_DOMAINS):
                result[i] = {"error": "Unsupported domain"}
                continue

            document = fetch_url(i)
            if not document:
                result[i] = {"error": "Failed to fetch content"}
            else:
                extracted_content = extract(document)
                if not extracted_content:
                    result[i] = {"error": "Extraction failed"}
                else:
                    result[i] = {"content": extracted_content}
        except Exception as e:
            result[i] = {"error": f"Exception: {str(e)}"}
    return {"extracted_content": result}
