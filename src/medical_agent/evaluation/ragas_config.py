"""
RAGAS evaluation configuration with Azure OpenAI wrappers.

Provides LLM and embedding wrappers compatible with RAGAS v0.4+
using existing Azure OpenAI credentials from application settings.
"""

import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# Load .env into os.environ so LANGCHAIN_* vars are available
load_dotenv()

from medical_agent.core.config import settings  # noqa: E402


def get_evaluator_llm(use_mini: bool = False) -> LangchainLLMWrapper:
    """
    Get a RAGAS-compatible LLM wrapper using Azure OpenAI.

    Args:
        use_mini: If True, uses gpt-4o-mini deployment for cheaper generation.

    Returns:
        LangchainLLMWrapper around AzureChatOpenAI
    """
    deployment = (
        settings.azure_openai_mini_deployment_name
        if use_mini
        else settings.azure_openai_deployment_name
    )

    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0,
    )

    return LangchainLLMWrapper(llm)


def get_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """
    Get a RAGAS-compatible embeddings wrapper using Azure OpenAI.

    Uses embedding-specific credentials with fallback to main Azure OpenAI
    credentials, matching the pattern in llamaindex_retrieval.py.

    Returns:
        LangchainEmbeddingsWrapper around AzureOpenAIEmbeddings
    """
    embed_api_key = settings.azure_openai_embedding_api_key or settings.azure_openai_api_key
    embed_endpoint = settings.azure_openai_embedding_endpoint or settings.azure_openai_endpoint
    embed_api_version = (
        settings.azure_openai_embedding_api_version or settings.azure_openai_api_version
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_openai_embedding_deployment_name,
        api_key=embed_api_key,
        azure_endpoint=embed_endpoint,
        api_version=embed_api_version,
    )

    return LangchainEmbeddingsWrapper(embeddings)


def setup_langsmith_tracing(project_suffix: str = "evaluation") -> None:
    """
    Configure LangSmith tracing environment variables for RAGAS evaluation.

    Sets up tracing so all RAGAS LLM calls are visible in the LangSmith dashboard.
    Reads from settings (LANGSMITH_API_KEY) or existing env var (LANGCHAIN_API_KEY).

    Args:
        project_suffix: Suffix appended to the base LangSmith project name.
            e.g., "evaluation" → "phera-agent-evaluation"
    """
    api_key = settings.langsmith_api_key or os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        return

    base_project = settings.langsmith_project or os.environ.get("LANGCHAIN_PROJECT", "phera-agent")

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = f"{base_project}-{project_suffix}"

    # Preserve LangSmith endpoint (e.g., EU region)
    endpoint = os.environ.get("LANGCHAIN_ENDPOINT", "")
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
