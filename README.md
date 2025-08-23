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


## Dev
- After installing new packages into the environment, make sure to do `pip freeze > requirements.txt`
- Note : OLLAMA integration not working yet, use GEMINI API KEY only.

## .env 

`
# can skip over Reddit credentials, haven't integrated them as tools yet.
REDDIT_USERNAME =
REDDIT_PASSWORD =
REDDIT_CLIENT_ID =
REDDIT_SECRET_KEY =
OLLAMA_MODEL=llama3.2:latest
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-1.5-flash
USE_OLLAMA=false
`
