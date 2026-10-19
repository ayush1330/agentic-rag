"""LlamaIndex Azure OpenAI embedding model factory.

Note: LLM generation uses LangChain (see agents/nodes.py).
This module only provides embedding models for retrieval.
"""

import logging
from typing import TYPE_CHECKING

from medical_agent.core.config import settings
from medical_agent.core.exceptions import LLMError

if TYPE_CHECKING:
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

logger = logging.getLogger(__name__)


def get_llama_index_embed_model() -> "AzureOpenAIEmbedding":
    """Get a LlamaIndex-compatible Azure OpenAI embedding model."""
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

    if not settings.is_azure_openai_embedding_configured():
        raise LLMError("Azure OpenAI embeddings are not configured")

    api_key = settings.azure_openai_embedding_api_key or settings.azure_openai_api_key
    endpoint = settings.azure_openai_embedding_endpoint or settings.azure_openai_endpoint
    api_version = settings.azure_openai_embedding_api_version or settings.azure_openai_api_version

    return AzureOpenAIEmbedding(
        model=settings.azure_openai_embedding_deployment_name,
        deployment_name=settings.azure_openai_embedding_deployment_name,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
