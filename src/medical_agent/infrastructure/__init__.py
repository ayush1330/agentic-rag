"""Infrastructure service clients.

Note: LLM generation now uses LangChain (see agents/nodes.py).
This module only exports embedding models for retrieval.
"""

from medical_agent.infrastructure.azure_openai import (
    get_llama_index_embed_model,
)
from medical_agent.infrastructure.gcp_storage import (
    GCPStorageClient,
    get_storage_client,
)

__all__ = [
    "GCPStorageClient",
    "get_storage_client",
    "get_llama_index_embed_model",
]
