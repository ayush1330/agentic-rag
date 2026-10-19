"""
Query refinement for agentic retrieval retry (Phase 3).

Generates semantically different queries when initial retrieval is insufficient.
Includes similarity checking to prevent loops (queries too similar or too different).
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity

from medical_agent.agents.state import MedicalAgentState
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


async def generate_refined_query(
    original_query: str,
    user_message: str,
    retrieved_docs: str,
    confidence_reason: str,
    llm: AzureChatOpenAI,
) -> str:
    """
    Generate a refined query based on what's missing from initial retrieval.

    Uses LLM to reason about retrieval failure and generate targeted query.

    Args:
        original_query: User's original search query
        user_message: Original user question
        retrieved_docs: Text of retrieved documents (why they failed)
        confidence_reason: Reasoning for low confidence
        llm: Azure OpenAI LLM instance

    Returns:
        Refined query string targeting missing aspects
    """
    logger.info("Generating refined query for retrieval retry")

    refinement_prompt = f"""You are a medical query refinement specialist. The initial retrieval for this patient query was insufficient. These queries search a database of peer-reviewed women's vaginal health research papers.

<original_query>{original_query}</original_query>
<user_question>{user_message}</user_question>

<retrieved_documents_summary>
{retrieved_docs[:1500]}
</retrieved_documents_summary>

<confidence_issue>{confidence_reason}</confidence_issue>

<instructions>
Generate a focused, refined query that:
1. Targets what the original query missed
2. Is semantically different from the original (but related)
3. Uses precise medical terminology
4. Focuses on one specific aspect the initial retrieval missed
5. Preserves the patient's relevant conditions (pH, diagnoses, symptoms) if they help target better results
</instructions>

<example>
Original: "Clinical significance of elevated vaginal pH in bacterial vaginosis with vulvar pruritus"
Retrieved docs covered BV diagnosis but not treatment → Refined: "Antibiotic treatment efficacy and recurrence prevention for bacterial vaginosis in symptomatic patients"
</example>

Do NOT repeat the original query, use vague language, or go off-topic.

Output ONLY the refined query string, nothing else:"""

    response = await llm.ainvoke([HumanMessage(content=refinement_prompt)])
    refined_query = str(response.content).strip()

    logger.info(f"Refined query: {refined_query[:100]}...")

    return refined_query


# ============================================================================
# ENHANCEMENT NOTE (Phase 3 Deviation from Spec)
# ============================================================================
# Spec: "Check query diversity with embedding similarity (>10% different)"
#       → Implies single 0.90 threshold (reject if > 90% similar)
#
# Implementation: Goldilocks zone with TWO thresholds (75-90%)
#       → Rejects BOTH too-similar AND too-different queries
#
# Reasoning:
#   1. Too similar (>90%): Refined query won't help retrieval (loops)
#   2. Too different (<75%): Refined query might go off-topic (loses context)
#   3. Goldilocks zone (75-90%): Different enough to find new results,
#      but similar enough to stay on-topic
#
# Impact: More robust retry loop that avoids both infinite loops AND
#         off-topic divergence. Empirically better for medical QA.
# ============================================================================


def check_query_similarity(
    original_query: str,
    refined_query: str,
    embed_model: AzureOpenAIEmbedding,
    lower_threshold: float = 0.75,
    upper_threshold: float = 0.90,
) -> bool:
    """
    Check if refined query is in the "Goldilocks zone" — different enough but not too different.

    Uses embedding similarity:
    - If similarity > upper_threshold (0.90): Too similar, skip retry
    - If similarity < lower_threshold (0.75): Too different, skip retry
    - If lower_threshold <= similarity <= upper_threshold: Just right, proceed with retry

    Args:
        original_query: Original search query
        refined_query: Newly generated refined query
        embed_model: Azure OpenAI embedding model
        lower_threshold: Minimum similarity threshold (default 0.75 = 75%)
        upper_threshold: Maximum similarity threshold (default 0.90 = 90%)

    Returns:
        bool: True if queries are in acceptable range (proceed with retry)
              False if too similar or too different (skip retry)
    """
    logger.info("Checking query similarity")

    # Get embeddings
    original_embedding = embed_model.get_text_embedding(original_query)
    refined_embedding = embed_model.get_text_embedding(refined_query)

    # Compute cosine similarity
    similarity = cosine_similarity(
        [original_embedding],
        [refined_embedding],
    )[0][0]

    logger.info(f"Query similarity: {similarity:.3f} (range: {lower_threshold}-{upper_threshold})")

    # Check both bounds
    if similarity > upper_threshold:
        logger.warning(
            f"Refined query too similar to original ({similarity:.3f} > {upper_threshold}). "
            f"Skipping retry to avoid loop."
        )
        return False

    if similarity < lower_threshold:
        logger.warning(
            f"Refined query too different from original ({similarity:.3f} < {lower_threshold}). "
            f"Skipping retry to avoid going off-topic."
        )
        return False

    logger.info(
        f"Query in acceptable range ({lower_threshold} <= {similarity:.3f} <= {upper_threshold}). "
        f"Proceeding with retry."
    )
    return True


async def refine_query_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Query refinement node: Generates refined query for retry when confidence is low.

    Process:
    1. LLM analyzes what's missing from retrieved documents
    2. Generate refined query targeting those missing aspects
    3. Check similarity to original query (Goldilocks zone: 75-90% similar)
    4. If in acceptable range, return refined query for retry
    5. If too similar or too different, mark to skip retry

    Args:
        state: Current agent state with docs_text, confidence info, messages

    Returns:
        State update with refined_query, retry_count, refinement_history, skip_retry
    """
    logger.info("Executing refine_query_node")

    original_query = state.get("original_query", "")
    user_message = ""

    # Extract user message
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        user_message = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message.get("content", ""))
        )

    docs_text = state.get("docs_text", "")
    confidence_score = state.get("confidence_score", 0.0)
    confidence_method = state.get("confidence_method", "unknown")

    # Create confidence reason explanation
    confidence_reason = f"Confidence score: {confidence_score:.2f}, method: {confidence_method}"

    # Get LLM
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    # Generate refined query
    refined_query = await generate_refined_query(
        original_query=original_query,
        user_message=user_message,
        retrieved_docs=docs_text,
        confidence_reason=confidence_reason,
        llm=llm,
    )

    # Check similarity
    embed_model = AzureOpenAIEmbedding(
        model=settings.azure_openai_embedding_deployment_name,
        deployment_name=settings.azure_openai_embedding_deployment_name,
        api_key=settings.azure_openai_embedding_api_key or settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_embedding_endpoint or settings.azure_openai_endpoint,
        api_version=settings.azure_openai_embedding_api_version
        or settings.azure_openai_api_version,
    )

    is_acceptable = check_query_similarity(
        original_query=original_query,
        refined_query=refined_query,
        embed_model=embed_model,
        lower_threshold=0.75,  # Minimum similarity
        upper_threshold=0.90,  # Maximum similarity
    )

    # Prepare state update
    current_retry_count = state.get("retry_count", 0)
    current_history = state.get("refinement_history", [])

    if is_acceptable:
        logger.info(f"Refined query approved for retry: {refined_query[:80]}...")

        return {
            "retry_count": current_retry_count + 1,
            "refinement_history": current_history + [refined_query],
            "skip_retry": False,  # Proceed with retry
        }
    else:
        logger.warning("Refined query outside acceptable range. Skipping retry.")

        return {
            "skip_retry": True,  # Skip retry, go straight to generate
        }
