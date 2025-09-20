from google.cloud import bigquery
import google.genai as genai
import pandas as pd
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
KEY_PATH = os.environ.get("KEY_PATH")
PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = os.environ.get("DATASET_ID")
TABLE_ID = os.environ.get("TABLE_ID")
GENAI_API_KEY = os.environ.get("GEMINI_API")

# -----------------------------
# Clients
# -----------------------------
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
genai_client = genai.Client(api_key=GENAI_API_KEY)

# -----------------------------
# Embedding function
# -----------------------------
def embed_text(texts, model="gemini-embedding-001"):
    if isinstance(texts, str):
        texts = [texts]
    response = genai_client.models.embed_content(model=model, contents=texts)
    embeddings = []
    for emb in response.embeddings:
        embeddings.append([float(v) for v in emb.values])
    return embeddings if len(embeddings) > 1 else embeddings[0]

# -----------------------------
# Create table if not exists
# -----------------------------
def create_table_if_not_exists():
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    try:
        bq_client.get_table(table_id)
        print(f"Table {table_id} already exists.")
    except:
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED")
        ]
        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table)
        print(f"Created table {table_id}")

# -----------------------------
# Push documents to BigQuery
# -----------------------------
def push_docs_to_bq(docs_df, model="gemini-embedding-001"):
    """
    Adds documents to the BigQuery table with embeddings.
    """
    create_table_if_not_exists()

    # Generate embeddings
    embeddings = embed_text(docs_df["text"].tolist(), model=model)
    docs_df["embedding"] = embeddings

    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    # Load DataFrame into BigQuery
    job = bq_client.load_table_from_dataframe(docs_df, table_id)
    job.result()
    print(f"Pushed {len(docs_df)} documents to BigQuery table {table_id}")

# -----------------------------
# Retrieve documents using VECTOR_SEARCH
# -----------------------------
def retrieve_from_bq(query_embedding, top_k=3):
    """
    Retrieve top-k documents using VECTOR_SEARCH on user-provided embeddings.
    """
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    embedding_column = "embedding"

    # Convert embedding vector to BigQuery array literal
    emb_array_literal = "[" + ",".join([str(float(x)) for x in query_embedding]) + "]"

    sql = f"""
    SELECT base.id AS id,
        base.text AS text,
        base.embedding AS embedding,
        distance
    FROM VECTOR_SEARCH(
        TABLE `{table_id}`,
        '{embedding_column}',
        (SELECT {emb_array_literal} AS {embedding_column}),
        top_k => {top_k},
        distance_type => 'COSINE'
    )
    """

    query_job = bq_client.query(sql)
    results = query_job.result()
    print(results)
    df= results.to_dataframe()
    print("Columns in retrieved DataFrame:", df.columns)
    return df



# -----------------------------
# Call language model
# -----------------------------
def call_language_model(prompt, model="gemini-chat-001"):
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash", contents="Explain how AI works in a few words"
    )
    return response.text

# -----------------------------
# RAG function
# -----------------------------
def rag_answer(user_query):
    query_emb = embed_text(user_query)
    docs = retrieve_from_bq(query_emb)
    if docs.empty:
        return "No relevant documents found."
    context = "\n".join(docs["text"].tolist())
    prompt = f"Answer the following question using only this context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    return call_language_model(prompt)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Sample documents
    docs = pd.DataFrame([
        {"id": "1", "text": "Solar energy is encouraged by the renewable energy policy."},
        {"id": "2", "text": "Wind energy projects are incentivized in the policy."},
        {"id": "3", "text": "Coal plants are gradually phased out."}
    ])

    # Push documents to BigQuery
    push_docs_to_bq(docs)

    # RAG query
    answer = rag_answer("What does the policy say about renewable energy?")
    print(answer)
