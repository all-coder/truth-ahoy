import os

from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

from dotenv import load_dotenv
from .prompts import return_instructions_root 

load_dotenv()

retrieval_agent = VertexAiRagRetrieval(
    name='retrieval_agent',
    description=(
        'Use this agent to retrieve factual information with truth values and sources from the knowledge base. '
        'Each result includes the fact, its truth value/explanation, source URL, and associated image if available.'
    ),
    rag_resources=[
        rag.RagResource(
            rag_vector_db=rag.RagVectorDb(
                rag_vector_db_config=rag.RagVectorDbConfig(
                    bigquery_config=rag.BigQueryConfig(
                        project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'),
                        dataset_id=os.environ.get('BIGQUERY_DATASET'),
                        table_id=os.environ.get('BIGQUERY_TABLE'),
                        content_column=os.environ.get('CONTENT_COLUMN', 'content'),
                        embedding_column=os.environ.get('EMBEDDING_COLUMN', 'embedding'),  # Optional
                        metadata_columns=os.environ.get('METADATA_COLUMNS', '').split(',') if os.environ.get('METADATA_COLUMNS') else ['truth_value', 'source', 'image_url']
                    )
                )
            )
        )
    ],
    similarity_top_k=10,
    vector_distance_threshold=0.5,
)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='rag_agent',
    instruction=return_instructions_root(),
    tools=[retrieval_agent],
)