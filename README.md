## Updates
- The Web Agent has been fully implemented.  
- We are adding a new feature to ensure responsible action: the tool will send emails or community posts to alert users about new threats or scams reported by others.  
  - A specialized database will be maintained to track reported threats and community alerts.  
- We will initially test the `Trafilatura` library for generic web scraping. Since most news articles are statically rendered, extracting content and image URLs should be straightforward. In the future, to handle dynamically rendered news portals more effectively, we plan to use a hybrid approach combining Playwright or Selenium with Trafilatura.
- OLLAMA integration with ADK is currently causing issues; use the GEMINI API key for the time being.
- We are using Tavily API as our primary search engine for now, it has a free tier of ~1000 monthly credits, we will go with it for now for testing paurposes. We will switch to google search in production / deployment.

## Note
- If installing new packages onto the environment, run `pip freeze > requirements.txt` to update the `requirements.txt` file.
- To test the Vision tools, enable the Cloud Vision API in the Google Cloud Console, then create a service account and download its key and write the filepath to the .env (default : `vision_gcp.json`)

## Setting Up

### 1. Create and Activate Virtual Environment

- On Windows:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

- On Mac/Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate Virtual environment
source venv/bin/activate
```

- Installing required python libraries

```bash
pip install -r requirements.txt
```

### Testing

```
# run this command at the project root
python backend/main.py

```

## .env

```
# Conditionals
USE_TAVILY=true
USE_OLLAMA=false

#REDDIT
REDDIT_USERNAME =
REDDIT_PASSWORD =
REDDIT_CLIENT_ID =
REDDIT_SECRET_KEY =
OLLAMA_MODEL=llama3.2:latest
GOOGLE_GENAI_USE_VERTEXAI=FALSE

#TAVILY KEY
TAVILY_API_KEY=

# ADK AGENTS USE THIS
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-1.5-flash

#SPECFICALLY FOR IMAGE ANALYSIS
GEMINI_IMAGE=

# SERVICE ACCOUNT KEY
GOOGLE_CLOUD_SERVICE_KEY=

# BIGQUERY
KEY_PATH = ""
PROJECT_ID = ""
DATASET_ID = ""
TABLE_ID = ""
GENAI_API_KEY =""

#VERTEX AI RAG RETRIEVAL 
GOOGLE_CLOUD_PROJECT=""
BIGQUERY_DATASET=""
BIGQUERY_TABLE=""
CONTENT_COLUMN=""
EMBEDDING_COLUMN=""
METADATA_COLUMNS=""
```

## Citations

```shell
@inproceedings{barbaresi-2021-trafilatura,
  title = {{Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction}},
  author = "Barbaresi, Adrien",
  booktitle = "Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
  pages = "122--131",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.acl-demo.15",
  year = 2021,}
```
