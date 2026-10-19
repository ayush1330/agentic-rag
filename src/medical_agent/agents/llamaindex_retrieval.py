"""
Medical RAG retrieval using LlamaIndex retriever.retrieve().

Provides structured node retrieval from medical research papers.
Returns NodeWithScore objects for use in LangGraph workflows.
"""

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llama_index.core.base.base_retriever import BaseRetriever


# ============================================================================
# Connection String Builder
# ============================================================================


def _build_vector_store_connection_strings() -> tuple[str, str]:
    """
    Build SQLAlchemy-compatible connection strings for PGVectorStore.

    For Cloud SQL Unix socket paths (host starts with /), uses @localhost:5432
    as a valid placeholder in the URL so SQLAlchemy can parse it without errors,
    and passes the actual socket path via ?host= query parameter. Both psycopg2
    and asyncpg drivers receive this query param and override the localhost value
    with the socket path, treating it as a Unix domain socket connection.

    For regular TCP connections, uses standard host:port URL format.
    """
    host = settings.resolved_postgres_host
    port = settings.resolved_postgres_port
    user = settings.resolved_postgres_user
    password = settings.resolved_postgres_password
    db = settings.resolved_postgres_db
    encoded_password = quote_plus(password)

    if host.startswith("/"):
        sync_url = (
            f"postgresql+psycopg2://{user}:{encoded_password}@localhost:5432/{db}?host={host}"
        )
        async_url = (
            f"postgresql+asyncpg://{user}:{encoded_password}@localhost:5432/{db}?host={host}"
        )
    else:
        sync_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{db}"
        async_url = f"postgresql+asyncpg://{user}:{encoded_password}@{host}:{port}/{db}"

    return sync_url, async_url


# ============================================================================
# Retriever Builder
# ============================================================================

# Module-level cache for retriever singleton
_retriever_cache: dict[int, "BaseRetriever"] = {}


def build_retriever(similarity_top_k: int = 5) -> "BaseRetriever":
    """
    Build or return cached retriever with vector store connection.

    Uses module-level caching to avoid rebuilding connections on every query.

    Args:
        similarity_top_k: Number of top chunks to retrieve

    Returns:
        Configured LlamaIndex BaseRetriever instance.
    """
    # Check cache first
    cache_key = similarity_top_k
    if cache_key in _retriever_cache:
        logger.debug(f"Using cached retriever for top_k={similarity_top_k}")
        return _retriever_cache[cache_key]

    logger.info(f"Building new retriever for top_k={similarity_top_k}")

    # Initialize embeddings with correct endpoint (separate embedding resource)
    embed_api_key = settings.azure_openai_embedding_api_key or settings.azure_openai_api_key
    embed_endpoint = settings.azure_openai_embedding_endpoint or settings.azure_openai_endpoint
    embed_api_version = (
        settings.azure_openai_embedding_api_version or settings.azure_openai_api_version
    )

    embed_model = AzureOpenAIEmbedding(
        model=settings.azure_openai_embedding_deployment_name,
        deployment_name=settings.azure_openai_embedding_deployment_name,
        api_key=embed_api_key,
        azure_endpoint=embed_endpoint,
        api_version=embed_api_version,
    )

    # Connect to vector store using pre-built connection strings that correctly
    # handle Cloud SQL Unix socket paths via the ?host= query parameter approach.
    sync_conn_str, async_conn_str = _build_vector_store_connection_strings()
    vector_store = PGVectorStore.from_params(
        connection_string=sync_conn_str,
        async_connection_string=async_conn_str,
        table_name="paper_chunks",
        embed_dim=settings.embedding_dimension,
        hybrid_search=True,
        text_search_config="english",
        perform_setup=False,
    )

    # Create VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Return retriever for structured node retrieval with hybrid search
    # vector_store_query_mode="hybrid" enables BM25 + vector fusion
    # alpha=0.3 means 70% BM25 (keyword) + 30% vector (semantic)
    # BM25-heavy is optimal for medical terminology precision
    retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        vector_store_query_mode="hybrid",
        alpha=0.3,  # BM25-heavy for medical terminology precision
    )

    # Cache for future use
    _retriever_cache[cache_key] = retriever
    logger.info(f"Retriever cached for top_k={similarity_top_k}")

    return retriever


def retrieve_nodes(query: str, similarity_top_k: int = 5) -> list[NodeWithScore]:
    """
    Retrieve medical research nodes for a query.

    Args:
        query: Search query
        similarity_top_k: Number of top chunks to retrieve

    Returns:
        List of NodeWithScore objects (nodes + relevance scores)
    """
    logger.info(f"Retrieving nodes for query: {query[:100]}...")
    retriever = build_retriever(similarity_top_k=similarity_top_k)
    nodes = retriever.retrieve(query)
    logger.info(f"Retrieved {len(nodes)} nodes")
    return nodes
