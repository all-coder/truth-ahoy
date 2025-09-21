"""Fact-check ETL pipeline.

This script fetches recent fact-check items from Google's Fact Check API,
parses and validates them with Pydantic models, embeds their content using
Google Generative AI, and stores the results directly into BigQuery (batch
ingestion with MERGE upsert). It is designed to run daily via cron with robust
duplicate handling and quota-aware embedding.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import requests
from google import genai
from google.genai.errors import ClientError
from pydantic import BaseModel, Field
from google.cloud import bigquery


GOOGLE_API_KEY_A = os.getenv("GOOGLE_API_KEY_A")
GOOGLE_API_KEY_B = os.getenv("GOOGLE_API_KEY_B")
GOOGLE_API_KEY_C = os.getenv("GOOGLE_API_KEY_C")

API_KEYS = [GOOGLE_API_KEY_A, GOOGLE_API_KEY_B, GOOGLE_API_KEY_C]
clients = [genai.Client(api_key=key) for key in API_KEYS]

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET = os.getenv("BQ_DATASET")
BIGQUERY_TABLE = os.getenv("BQ_TABLE")
BIGQUERY_STAGING_TABLE = os.getenv("BQ_STAGING_TABLE")

# Create a single BigQuery client
try:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
except Exception as _e:  # Defer hard failure until first use
    bq_client = None

# Final table schema (matches your BigQuery table screenshot)
BQ_SCHEMA = [
    bigquery.SchemaField("content", "STRING"),
    bigquery.SchemaField("source_url", "STRING"),
    bigquery.SchemaField("truth_val", "STRING"),
    bigquery.SchemaField("spread_vectors", "STRING"),
    bigquery.SchemaField("image_url", "STRING"),
    bigquery.SchemaField("content_hash", "STRING"),
    bigquery.SchemaField("insertion_time", "DATETIME"),
    bigquery.SchemaField("published_time", "DATETIME"),
    bigquery.SchemaField("claim_date", "DATETIME"),
    bigquery.SchemaField("publisher_name", "STRING"),
    bigquery.SchemaField("author", "STRING"),
    bigquery.SchemaField("country", "STRING"),
    bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
    bigquery.SchemaField("labels", "JSON"),
    bigquery.SchemaField("claim_appearances", "JSON"),
    bigquery.SchemaField("canonical_url", "JSON"),
]


def _ensure_dataset_and_tables():
    """Create dataset and tables if they don't exist."""
    global bq_client
    if bq_client is None:
        # Attempt late initialization to allow env to be set after import
        bq_client = bigquery.Client(project=GCP_PROJECT_ID)

    dataset_ref = bigquery.Dataset(f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}")
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        bq_client.create_dataset(dataset_ref, exists_ok=True)

    # Ensure final table exists (cluster by content_hash to reduce bytes for point lookups)
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
    try:
        bq_client.get_table(table_id)
    except Exception:
        tbl = bigquery.Table(table_id, schema=BQ_SCHEMA)
        try:
            tbl.clustering_fields = ["content_hash"]
        except Exception:
            pass
        bq_client.create_table(tbl, exists_ok=True)

    # Ensure staging table exists (we truncate per batch load; also cluster)
    staging_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_STAGING_TABLE}"
    try:
        bq_client.get_table(staging_id)
    except Exception:
        st = bigquery.Table(staging_id, schema=BQ_SCHEMA)
        try:
            st.clustering_fields = ["content_hash"]
        except Exception:
            pass
        bq_client.create_table(st, exists_ok=True)



def get_facts(count: int = 50, offset: int = 0) -> str:
    """Fetch a page of fact-check items from the API with retries.

    Parameters:
        count: Number of results to fetch.
        offset: Offset for pagination.

    Returns:
        The raw JSON (as a string) returned by the API (prefix removed).

    Notes:
        - Implements capped exponential backoff and keeps retrying until a
          successful HTTP response is obtained.
        - Uses a realistic browser-like User-Agent and sensible timeouts.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8,hi;q=0.7"
    }

    url = (
        f"https://toolbox.google.com/factcheck/api/search?hl=en&num_results={count}"
        f"&force=false&offset={offset}&query=list%3Arecent"
    )
    attempt = 0
    backoff = 1.0
    while True:
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.text[6:]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            attempt += 1
            print(f"HTTP fetch error (attempt {attempt}): {e}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            # capped exponential backoff up to ~60s
            backoff = min(backoff * 2, 60)
            continue
        except requests.exceptions.HTTPError as he:
            code = he.response.status_code if he.response is not None else 'unknown'
            print(f"HTTP {code} error on fetch: {he}. Retrying in {backoff:.1f}s...")
            attempt += 1
            time.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)
            continue


class Fact(BaseModel):
    """Structured representation of a fact-check item suitable for persistence."""

    content: str
    source_url: Optional[str] = None
    truth_val: Optional[str] = None
    embedding: Optional[List[float]] = None
    spread_vectors: Optional[str] = None
    image_url: Optional[str] = None
    content_hash: str = Field(..., description="SHA-256 hash of content")
    labels: dict[str, float] = Field(default_factory=dict)
    insertion_time: str
    published_time: Optional[str] = None
    claim_date: Optional[str] = None
    publisher_name: Optional[str] = None
    author: Optional[str] = None
    country: Optional[str] = None
    # Keep as native Python (BigQuery JSON will accept dict/list directly)
    claim_appearances: Optional[List[dict]] = None
    canonical_url: Optional[List[str]] = None


def parse_facts(fact_data: str) -> List[Fact]:
    """Parse raw API JSON into validated Fact models.

    Parameters:
        fact_data: The JSON payload string returned by get_facts.

    Returns:
        A list of validated Fact instances.
    """
    data = json.loads(fact_data)
    response_data = data[0]
    
    if response_data[0] == "er":
        print("Error in fetching data")
        return []
    
    label_id_to_name_list = response_data[2]
    label_map = dict(label_id_to_name_list)

    claim_items = response_data[1]
    
    collected_facts: List[Fact] = []
    for item in claim_items:
        edata: dict = {}

        edata["image_url"] = item[1]
        claim_data = item[0]

        edata["content"] = claim_data[0]
        edata["spread_vectors"] = claim_data[1][0] if claim_data[1] else None
        
        # Extract claim date
        claim_ts = claim_data[2]
        edata["claim_date"] = datetime.fromtimestamp(claim_ts).strftime("%Y-%m-%d %H:%M:%S") if claim_ts else None

        # Extract all verification data from the first fact-checker
        verification_data = claim_data[3]
        if verification_data:
            first_verification = verification_data[0]
            publisher_info = first_verification[0]
            
            edata["source_url"] = first_verification[1]
            edata["truth_val"] = first_verification[3]
            published_ts = first_verification[2]
            edata["published_time"] = datetime.fromtimestamp(published_ts).strftime("%Y-%m-%d %H:%M:%S") if published_ts else None
            
            # Publisher and author details
            edata["publisher_name"] = publisher_info[0]
            author_list = publisher_info[2]
            edata["author"] = author_list[0] if author_list and author_list[0] else None
            edata["country"] = publisher_info[4]
        else:
            # Set defaults if no verification is found
            edata.update({
                "source_url": None, "truth_val": "N/A", "published_time": None,
                "publisher_name": None, "author": None, "country": None
            })
        
        # Extract claim appearances
        appearances_data = claim_data[4]
        if appearances_data:
            edata["claim_appearances"] = [
                {"domain": appearance[0][1], "url": appearance[1]}
                for appearance in appearances_data
                if appearance and len(appearance) > 1 and appearance[0]
            ]
        else:
            edata["claim_appearances"] = None

        edata["insertion_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        edata["content_hash"] = hashlib.sha256(edata["content"].encode('utf-8')).hexdigest()
        
        # related urls
        if len(claim_data) > 12:
            canonical_url_list = claim_data[12]
            edata["canonical_url"] = canonical_url_list if canonical_url_list else None
        else:
            edata["canonical_url"] = None

        # Corrected label extraction
        if len(claim_data) > 8 and claim_data[8]:
            label_items = claim_data[8]
            edata["labels"] = {
                label_map[label_id]: float(score)
                for label_id, score, *_ in label_items
                if label_id in label_map
            }
        else:
            edata["labels"] = {}

        # Validate and coerce using Pydantic models
        try:
            fact = Fact(
                content=edata["content"],
                source_url=edata.get("source_url"),
                truth_val=edata.get("truth_val"),
                spread_vectors=edata.get("spread_vectors"),
                image_url=edata.get("image_url"),
                content_hash=edata["content_hash"],
                labels=edata.get("labels", {}),
                insertion_time=edata["insertion_time"],
                published_time=edata.get("published_time"),
                claim_date=edata.get("claim_date"),
                publisher_name=edata.get("publisher_name"),
                author=edata.get("author"),
                country=edata.get("country"),
                claim_appearances=edata.get("claim_appearances"),
                canonical_url=edata.get("canonical_url"),
            )
        except Exception as e:
            print(f"Validation error building Fact: {e}")
            continue

        collected_facts.append(fact)
    return collected_facts


def batch_embed_queries(queries: Sequence[str]) -> List[Iterable[float]]:
    """Embed a sequence of text queries using Google GenAI with quota handling.

    Parameters:
        queries: Sequence of strings to embed.

    Returns:
        A list of embedding vectors (as iterables of floats) in the same order
        as the input queries.

    Behavior:
        - Rotates among multiple API clients on RPM (per-minute) limits.
        - Marks a client exhausted on RPD (per-day) limits and never reuses it
          for the remainder of the run. Stops entirely when all are exhausted.
    """
    embeddings: List[Iterable[float]] = []
    batch_size = 100
    client_idx = 0
    client_rpd_exhausted = [False] * len(clients)
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        while True:
            # If all clients have hit RPD, stop script
            if all(client_rpd_exhausted):
                print("All clients have hit RPD (daily limit). Stopping script.")
                raise SystemExit(0)
            # Find the next non-RPD-exhausted client (rotate after every embedding)
            next_idx = get_next_available_client_idx(client_rpd_exhausted, client_idx) 
            if next_idx is None:
                print("No available clients (all RPD exhausted). Stopping script.")
                raise SystemExit(0)
            client_idx = next_idx
            client = clients[client_idx]
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch
                )
                embeddings.extend(result.embeddings)
                print("embedded", len(embeddings), "queries in this batch (client", client_idx, ")")
                break
            except ClientError as ce:
                error_str = str(ce).lower()
                # RPD (daily) error string
                if "embedcontentrequestsperdayperuserperprojectpermodel-freetier" in error_str:
                    print(f"RPD (daily) limit reached for client {client_idx}. Marking as exhausted.")
                    client_rpd_exhausted[client_idx] = True
                    # Move to next available client
                    next_idx = get_next_available_client_idx(client_rpd_exhausted, client_idx)
                    if next_idx is None:
                        print("No available clients (all RPD exhausted). Stopping script.")
                        raise SystemExit(0)
                    client_idx = next_idx
                    continue
                # RPM (minute) error string
                if (hasattr(ce, 'status_code') and ce.status_code == 429) or \
                   ("quota" in error_str or "rate" in error_str or "resource_exhausted" in error_str or "embedcontentrequestsperminuteperuserperprojectpermodel-freetier" in error_str):
                    print(error_str)
                    print(f"Quota or rate limit (RPM) exceeded for client {client_idx}. Switching to next client...")
                    # Move to next available client (skip RPD exhausted)
                    next_idx = get_next_available_client_idx(client_rpd_exhausted, client_idx)
                    if next_idx is None:
                        print("No available clients (all RPD exhausted). Stopping script.")
                        raise SystemExit(0)
                    if next_idx == client_idx:
                        print("All clients exhausted for RPM. Waiting 30 seconds before retrying...")
                        time.sleep(30)
                    client_idx = next_idx
                    continue
                else:
                    print(f"ClientError: {ce}. Retrying this batch after 3 seconds...")
                    time.sleep(3)
            except Exception as e:
                print(f"Embedding error: {e}. Retrying this batch after 3 seconds...")
                time.sleep(3)
    return embeddings

# Utility: get next available client index (not RPD exhausted)
def get_next_available_client_idx(client_rpd_exhausted: Sequence[bool], start_idx: int) -> Optional[int]:
    """Find the next client index not marked as RPD-exhausted.

    Returns None if all clients are exhausted.
    """
    n = len(client_rpd_exhausted)
    idx = (start_idx + 1) % n
    for _ in range(n):
        if not client_rpd_exhausted[idx]:
            return idx
        idx = (idx + 1) % n
    return None

def save_to_bigquery(
    facts: Sequence[Fact], embeddings: Sequence[Iterable[float] | dict | object]
) -> None:
    """Batch load to staging and MERGE into final BigQuery table."""
    if not facts:
        return

    rows: List[dict] = []
    for fact, emb in zip(facts, embeddings):
        flat = dict(emb)["values"]
        # JSON fields should be Python objects for BigQuery JSON type
        labels_json = fact.labels or {}
        claim_apps = fact.claim_appearances if fact.claim_appearances else None
        canonical = fact.canonical_url if fact.canonical_url else None

        rows.append(
            {
                "content": fact.content,
                "source_url": fact.source_url,
                "truth_val": fact.truth_val,
                "spread_vectors": fact.spread_vectors,
                "image_url": fact.image_url,
                "content_hash": fact.content_hash,
                "insertion_time": fact.insertion_time,
                "published_time": fact.published_time,
                "claim_date": fact.claim_date,
                "publisher_name": fact.publisher_name,
                "author": fact.author,
                "country": fact.country,
                "embedding": flat,
                "labels": labels_json,
                "claim_appearances": claim_apps,
                "canonical_url": canonical,
            }
        )

    staging_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_STAGING_TABLE}"
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"

    # Truncate staging table
    bq_client.query(f"TRUNCATE TABLE `{staging_id}`").result()

    load_job = bq_client.load_table_from_json(
        rows,
        staging_id,
        job_config=bigquery.LoadJobConfig(schema=BQ_SCHEMA, write_disposition="WRITE_APPEND"),
    )
    load_job.result()

    merge_sql = f"""
    MERGE `{table_id}` AS T
    USING `{staging_id}` AS S
    ON T.content_hash = S.content_hash
    WHEN MATCHED THEN UPDATE SET
      content = S.content,
      source_url = S.source_url,
      truth_val = S.truth_val,
      spread_vectors = S.spread_vectors,
      image_url = S.image_url,
      labels = S.labels,
      insertion_time = S.insertion_time,
      published_time = S.published_time,
      claim_date = S.claim_date,
      publisher_name = S.publisher_name,
      author = S.author,
      country = S.country,
      embedding = S.embedding,
      claim_appearances = S.claim_appearances,
      canonical_url = S.canonical_url
    WHEN NOT MATCHED THEN INSERT (
      content, source_url, truth_val, spread_vectors, image_url, content_hash,
      insertion_time, published_time, claim_date, publisher_name, author, country,
      embedding, labels, claim_appearances, canonical_url
    ) VALUES (
      S.content, S.source_url, S.truth_val, S.spread_vectors, S.image_url, S.content_hash,
      S.insertion_time, S.published_time, S.claim_date, S.publisher_name, S.author, S.country,
      S.embedding, S.labels, S.claim_appearances, S.canonical_url
    )
    """
    bq_client.query(merge_sql).result()

# --- New (BigQuery): Check if hash exists ---
def hash_exists(content_hash: str) -> bool:
    """Check if a given content_hash already exists in BigQuery."""
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
    query = f"SELECT 1 FROM `{table_id}` WHERE content_hash = @h LIMIT 1"
    job = bq_client.query(query, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("h", "STRING", content_hash)]
    ))
    rows = list(job.result())
    return len(rows) > 0

def hashes_exist_batch(hashes: List[str]) -> set[str]:
    """Return the subset of hashes that exist, using UNNEST to minimize cost."""
    if not hashes:
        return set()
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
    query = f"""
    WITH in_hashes AS (
      SELECT h FROM UNNEST(@hashes) AS h
    )
    SELECT T.content_hash AS h
    FROM `{table_id}` AS T
    INNER JOIN in_hashes I ON I.h = T.content_hash
    """
    job = bq_client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hashes", "STRING", hashes)]
        ),
    )
    return {row["h"] for row in job.result()}

"""Ingest metadata helpers"""
# Total items ingested so far (at start of run) used as contiguous skip window size
def get_total_ingested_count(default_hint: int = 0) -> int:
    """Return total count of rows in BigQuery facts table, or default_hint on error.

    Note: This uses a COUNT(*) query. For strictly free-tier usage, consider
    caching counts or relying on metadata where available (table.num_rows via
    API is also a metadata read and typically very cheap), but may lag until
    streaming buffers are committed.
    """
    try:
        table_ref = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
        table = bq_client.get_table(table_ref)
        return int(getattr(table, "num_rows", 0))
    except Exception as e:
        print(f"Count error (using default {default_hint}): {e}")
        return default_hint

# Embed a list of facts and persist
def embed_and_save(facts_list: Sequence[Fact]) -> int:
    """Embed a list of Fact models and persist them to BigQuery."""
    if not facts_list:
        return 0
    queries = [f.content for f in facts_list]
    embeddings = batch_embed_queries(queries)
    save_to_bigquery(facts_list, embeddings)
    return len(facts_list)


offset: int = 0
count: int = 200
total_inserted: int = 0

# Determine the size of the already-ingested contiguous block at start
_ensure_dataset_and_tables()
ingested_count_at_start = get_total_ingested_count(default_hint=0)
print(f"Total ingested at start (skip window): {ingested_count_at_start}")

# Phase 1: ingest newest until we hit the first known item, then skip the previous-day window
phase = 1  # 1=news scanning, 2=scan for first non-duplicate after skip, 3=bulk ingest without dup checks

try:
    while True:
        print(f"Fetching facts with offset {offset}")
        raw = get_facts(count=count, offset=offset)
        print("Fetched")
        facts = parse_facts(raw)
        if not facts:
            break

        if phase == 1:
            new_facts = []
            hit_duplicate = False
            # Batch check which hashes exist to minimize queries
            batch_hashes = [f.content_hash for f in facts]
            existing = hashes_exist_batch(batch_hashes)
            for fact in facts:
                if fact.content_hash in existing:
                    hit_duplicate = True
                    break
                new_facts.append(fact)

            if new_facts:
                print(f"Phase 1: embedding+saving {len(new_facts)} new facts before first duplicate")
                total_inserted += embed_and_save(new_facts)

            if hit_duplicate:
                print(f"Phase 1: first duplicate encountered. Skipping next {ingested_count_at_start} items.")
                offset += ingested_count_at_start
                phase = 2
                continue
            else:
                offset += count
                continue

        if phase == 2:
            # Find first non-duplicate; then process rest and switch to phase 3
            first_new_idx = None
            existing = hashes_exist_batch([f.content_hash for f in facts])
            for idx, fact in enumerate(facts):
                if fact.content_hash not in existing:
                    first_new_idx = idx
                    break
            if first_new_idx is None:
                # still duplicates here, keep scanning forward
                print("Phase 2: batch fully duplicate, continuing scan...")
                offset += count
                continue
            # Found the first non-duplicate after the skip window
            rest = facts[first_new_idx:]
            print(f"Phase 2: found first non-duplicate at index {first_new_idx}; processing {len(rest)} and switching to phase 3")
            total_inserted += embed_and_save(rest)
            phase = 3
            offset += count
            continue

        if phase == 3:
            # Guaranteed new territory per your assumption—no duplicate checks
            print(f"Phase 3: bulk embedding+saving {len(facts)} facts without duplicate checks")
            total_inserted += embed_and_save(facts)
            offset += count
            continue

except SystemExit:
    # Allow the RPD/RPM logic to stop the run but still print stats
    print("Run stopped due to quota limits.")
finally:
    print(f"Total inserted this run: {total_inserted}")
    # No explicit BigQuery cleanup required

