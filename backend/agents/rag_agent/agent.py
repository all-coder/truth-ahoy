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
        'Use this agent to retrieve documentation and reference materials from the RAG corpus,'
    ),
    rag_resources=[
        rag.RagResource(
            rag_corpus=os.environ.get('RAG_CORPUS')
        )
    ],
    similarity_top_k=5,
    vector_distance_threshold=0.6,
)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='rag_agent',
    instruction=return_instructions_root(),
    tools=[retrieval_agent],
)