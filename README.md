# 

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

## Testing
```
# run this command at the project root
python backend/main.py
```

## Dev
- After installing new packages into the environment, make sure to do `pip freeze > requirements.txt`
- Ensure that every tool intended for use by an agent or within a workflow includes a well-annotated docstring. Proper docstrings are essential for agents to understand and correctly utilize the tool

## Important Notes
- OLLAMA integration not working yet, use GEMINI API KEY only.
- We are using Tavily API as our primary search engine for now, it has a free tier of 1000 monthly credits, we will go with it for now for testing and dev purposes. We will switch to google search in production / deployment.

## .env 
```
REDDIT_USERNAME =
REDDIT_PASSWORD =
REDDIT_CLIENT_ID =
REDDIT_SECRET_KEY =
OLLAMA_MODEL=llama3.2:latest
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-1.5-flash
TAVILY_API_KEY=""

# conditionals
USE_TAVILY=true
USE_OLLAMA=false
```
