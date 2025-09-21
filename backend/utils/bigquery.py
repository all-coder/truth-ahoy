from google.cloud import bigquery
import google.genai as genai
import pandas as pd
from google.oauth2 import service_account
from dotenv import load_dotenv
import os
from google.cloud.bigquery import SchemaField as TableSchemaField
from typing import List, Union

load_dotenv()

KEY_PATH = os.environ.get("KEY_PATH")
PROJECT_ID = os.environ.get("PROJECT_ID")
TABLE_ID= os.environ.get("TABLE_ID")
GENAI_API_KEY = os.environ.get("GEMINI_API")

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
genai_client = genai.Client(api_key=GENAI_API_KEY)


TABLE_SCHEMA = [
    TableSchemaField(name="id", field_type="STRING", mode="REQUIRED"),
    TableSchemaField(name="user_claim", field_type="STRING", mode="REQUIRED"),
    TableSchemaField(name="image_links", field_type="STRING", mode="REPEATED"),
    TableSchemaField(name="web_sources", field_type="JSON", mode="NULLABLE"),
    TableSchemaField(name="web_analysis", field_type="STRING", mode="NULLABLE"),
    TableSchemaField(name="final_analysis", field_type="STRING", mode="REQUIRED"),
    TableSchemaField(name="deepfake_analysis", field_type="STRING", mode="NULLABLE"),
    TableSchemaField(name="image_analysis", field_type="STRING", mode="NULLABLE"),
    TableSchemaField(name="source_link", field_type="STRING", mode="NULLABLE"),
    TableSchemaField(name="embedding", field_type="FLOAT64", mode="REPEATED"),
]


def get_bigquery_schema() -> List[bigquery.SchemaField]:
    """Returns BigQuery schema from TABLE_SCHEMA definition."""
    return [bigquery.SchemaField(f.name, f.field_type, mode=f.mode) for f in TABLE_SCHEMA]

def create_table_if_not_exists():
    """Creates the BigQuery table if it does not exist using TABLE_SCHEMA."""
    table_id = table_id
    try:
        bq_client.get_table(table_id)
    except:
        table = bigquery.Table(table_id, schema=get_bigquery_schema())
        bq_client.create_table(table)

def embed_text(texts: Union[str, List[str]], model="gemini-embedding-001"):
    """Generates embeddings for a list of texts or a single text using Gemini embeddings."""
    if isinstance(texts, str):
        texts = [texts]
    response = genai_client.models.embed_content(model=model, contents=texts)
    embeddings = [[float(v) for v in emb.values] for emb in response.embeddings]
    return embeddings if len(embeddings) > 1 else embeddings[0]

def push_docs_to_bq(docs_df: pd.DataFrame, model="gemini-embedding-001"):
    """Pushes documents with embeddings to BigQuery table."""
    create_table_if_not_exists()
    embeddings = embed_text(docs_df["text"].tolist(), model=model)
    docs_df["embedding"] = [embeddings]
    table_id = TABLE_ID
    job = bq_client.load_table_from_dataframe(docs_df, table_id)
    job.result()

def retrieve_from_bq(user_query: str, top_k=3) -> pd.DataFrame:
    query_embedding = embed_text(user_query)
    """Retrieves top-k documents from BigQuery using VECTOR_SEARCH on embeddings."""
    table_id = TABLE_ID
    emb_array_literal = "[" + ",".join([str(float(x)) for x in query_embedding]) + "]"
    sql = f"""
    SELECT base.id AS id,
        base.text AS text,
        base.embedding AS embedding,
        distance
    FROM VECTOR_SEARCH(
        TABLE `{table_id}`,
        'embedding',
        (SELECT {emb_array_literal} AS embedding),
        top_k => {top_k},
        distance_type => 'COSINE'
    )
    """
    results = bq_client.query(sql).result()
    return results.to_dataframe()

def call_language_model(prompt: str, model="gemini-chat-001"):
    """Calls the Gemini language model to generate text for a given prompt."""
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text