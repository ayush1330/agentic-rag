"""
LangGraph node functions for medical RAG workflow.

Each node is a pure function that takes state and returns state updates.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from medical_agent.agents.llamaindex_retrieval import retrieve_nodes
from medical_agent.agents.prompts import (
    HIGH_CONFIDENCE_PROMPT,
    LOW_CONFIDENCE_PROMPT,
    MEDIUM_CONFIDENCE_PROMPT,
)
from medical_agent.agents.reranker import rerank_nodes_with_metadata
from medical_agent.agents.state import MedicalAgentState
from medical_agent.agents.utils import build_health_context, format_retrieved_nodes
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


class MedicalResponse(BaseModel):
    """Structured output schema for medical RAG responses."""

    overview: str = Field(
        description=(
            "Markdown bullet list summarizing the response. Use the bullet structure defined in "
            "the system prompt — do not invent a different format. Every claim must be cited with "
            "the same [N] markers used in the deep-dive fields — do not omit citations. Numbering "
            "is shared with the deep-dive fields: the same paper uses the same [N] in every field. "
            "Produce the deep-dive fields first, then summarize faithfully into overview."
        )
    )
    your_ph: str = Field(
        description=(
            "REQUIRED. Carries the 'Your pH' section content only. Write the section's content "
            "only — do not include a `## header` line. The field name is the section. "
            "Provide a clinical explanation of the pH value tied to the typical healthy range "
            "(3.8-4.5) and what shifts it. Every factual claim MUST be cited with [N] markers "
            "from <medical_evidence>. Do not include information not found in the documents."
        )
    )
    your_symptoms: str | None = Field(
        default=None,
        description=(
            "OPTIONAL. Set to null when the patient reported no symptoms (no-profile branch or "
            "bare pH-only request). When present, interpret the symptoms the patient reported "
            "(or the absence of them) and end with a '**Watch for:**' line listing symptoms that "
            "would warrant follow-up if they appeared later. Write the section's content only — "
            "do not include a `## header` line. The field name is the section. Every factual "
            "claim cited with [N] markers from <medical_evidence>."
        ),
    )
    your_personal_baseline: str | None = Field(
        default=None,
        description=(
            "OPTIONAL. Set to null when no profile was provided. When present, interpret age, "
            "ethnicity, and cycle phase against this pH. Write the section's content only — do "
            "not include a `## header` line. The field name is the section. Every factual claim "
            "cited with [N] markers from <medical_evidence>."
        ),
    )
    your_health_context: str | None = Field(
        default=None,
        description=(
            "OPTIONAL. Set to null when no profile was provided. When present, interpret reported "
            "medications and diagnoses (or note their absence and what that means for the "
            "result). Write the section's content only — do not include a `## header` line. The "
            "field name is the section. Every factual claim cited with [N] markers from "
            "<medical_evidence>."
        ),
    )
    next_steps: str = Field(
        description=(
            "REQUIRED. Concrete action: retest schedule, what to watch, when to see a doctor. "
            "This is the ONLY field where doctor / consultation / referral language is allowed. "
            "Write the section's content only — do not include a `## header` line. The field "
            "name is the section. Every factual claim cited with [N] markers from "
            "<medical_evidence>."
        )
    )
    used_citations: list[int] = Field(
        description=(
            "List of citation numbers actually used across the deep-dive fields + overview. "
            "For self-check only — backend renumbering pass treats text markers as authoritative."
        )
    )


def retrieve_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Retrieval node: Fetches relevant medical research chunks.

    Process:
        1. Extract user's latest message and save as original_query
        2. Use refined query from refinement_history if available (retry), else use message query (first call)
        3. Over-retrieve with raw query (no health context dilution)
        4. Rerank using cross-encoder + metadata weighting
        5. Format nodes as citation text with [1][2] markers
        6. Return docs_text + citations + original_query for next node

    Args:
        state: Current agent state with messages and health context

    Returns:
        State update with docs_text, citations, and original_query
    """
    logger.info("Executing retrieve_node")

    # Get the latest user message
    if not state.get("messages"):
        logger.warning("No messages in state")
        return {
            "docs_text": "No query provided.",
            "citations": [],
            "original_query": "",
        }

    # Extract message content (this is the original user query)
    last_message = state["messages"][-1]
    message_query = (
        last_message.content
        if hasattr(last_message, "content")
        else str(last_message.get("content", ""))
    )

    # Determine which query to use: refined (retry) or original (first call)
    refinement_history = state.get("refinement_history", [])
    if refinement_history:
        # Retry: use the last refined query
        user_query = refinement_history[-1]
        logger.info(
            f"Using refined query (retry {state.get('retry_count', 0)}): {user_query[:100]}..."
        )
    else:
        # First call: use the message query
        user_query = message_query
        logger.info(f"Retrieving nodes for query: {user_query[:100]}...")

    # Over-retrieve for reranking (15 candidates)
    nodes = retrieve_nodes(query=user_query, similarity_top_k=15)

    # Extract user health profile from state for metadata-weighted reranking
    health_profile = state.get("health_profile", {})

    # Rerank with cross-encoder (70%) + metadata overlap (30%)
    nodes = rerank_nodes_with_metadata(
        query=user_query,
        nodes=nodes,
        user_profile=health_profile,
        top_k=5,
    )

    # Format nodes into citation text
    docs_text, citations = format_retrieved_nodes(nodes)

    logger.info(f"Retrieved {len(citations)} citations after reranking")

    return {
        "docs_text": docs_text,
        "citations": citations,
        "original_query": message_query,  # Always save the user's message as original_query
    }


async def generate_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Generation node: Produces medical response with confidence-adaptive prompting.

    Uses display_query (user-perspective) instead of the clinical retrieval query
    to avoid third-person tone contamination. Splits System/Human messages for
    better instruction following.

    Three-tier prompt strategy based on retrieval confidence:
    - HIGH (>=0.75): Confident, direct, warm tone
    - MEDIUM (0.50-0.75): Measured, exploratory, still helpful
    - LOW (<0.50): Gentle, honest about limitations

    Args:
        state: Current agent state with docs_text, confidence_score, conversation history

    Returns:
        State update with new assistant message and used_citations
    """
    logger.info("Executing generate_node with confidence-adaptive prompting")

    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    structured_llm = llm.with_structured_output(MedicalResponse)

    docs_text = state.get("docs_text", "")
    ph_value = state.get("ph_value", 0.0)
    health_profile = state.get("health_profile", {})
    health_context = build_health_context(ph_value, health_profile)
    confidence_score = state.get("confidence_score", 0.5)

    # Use display_query (user-perspective) for the generation prompt.
    # Falls back to messages[-1] for follow-ups where display_query wasn't set.
    display_query = state.get("display_query", "")
    if not display_query:
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            display_query = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message.get("content", ""))
            )

    # Build conversation history with XML tags
    messages = state.get("messages", [])
    conversation_block = ""
    if len(messages) > 1:
        history_parts = []
        for msg in messages[:-1]:
            role = (
                msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "type", "user")
            )
            content = (
                msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            )
            history_parts.append(f"{role.capitalize()}: {content}")
        conversation_block = (
            "\n\n<conversation_history>\n" + "\n".join(history_parts) + "\n</conversation_history>"
        )

    # Select prompt tier based on confidence score
    if confidence_score >= 0.75:
        base_prompt = HIGH_CONFIDENCE_PROMPT
        prompt_tier = "HIGH"
    elif confidence_score >= 0.50:
        base_prompt = MEDIUM_CONFIDENCE_PROMPT
        prompt_tier = "MEDIUM"
    else:
        base_prompt = LOW_CONFIDENCE_PROMPT
        prompt_tier = "LOW"

    logger.info(f"Using {prompt_tier} confidence prompt (score: {confidence_score:.3f})")

    # Format the selected prompt with context (used as the user message)
    user_prompt = base_prompt.format(
        ph_value=ph_value,
        health_context=health_context,
        docs_text=docs_text,
        query=display_query,
    )

    if conversation_block:
        user_prompt += conversation_block

    logger.info("Generating structured response with Azure OpenAI (LangChain)")

    result: MedicalResponse = await structured_llm.ainvoke(  # type: ignore[assignment]
        [
            SystemMessage(
                content=(
                    "You are pHera, a women's health assistant. "
                    "Always address the patient directly using 'you' and 'your'. "
                    "Never refer to the patient in third person. "
                    "Ground every claim in the provided documents and cite with [1], [2], etc. "
                    "Do not add medical knowledge beyond the documents."
                )
            ),
            HumanMessage(content=user_prompt),
        ]
    )

    logger.info(
        f"Response generated (tier: {prompt_tier}, citations: {len(result.used_citations)})"
    )

    # Concatenate the 5 deep-dive fields (skipping None / empty) for the
    # AIMessage.content. LangGraph requires messages[-1].content to be a
    # string; this concat satisfies that. It is internal only — never sent
    # on the wire (the route layer reads the 5 fields directly from state).
    deep_dive_concat = "\n\n".join(
        s
        for s in (
            result.your_ph,
            result.your_symptoms,
            result.your_personal_baseline,
            result.your_health_context,
            result.next_steps,
        )
        if s
    )

    assistant_msg = AIMessage(
        content=deep_dive_concat,
        metadata={
            "confidence_score": confidence_score,
            "confidence_tier": prompt_tier,
        },
    )

    return {
        "messages": [assistant_msg],
        "used_citations": result.used_citations,
        "overview": result.overview,
        "your_ph": result.your_ph,
        "your_symptoms": result.your_symptoms,
        "your_personal_baseline": result.your_personal_baseline,
        "your_health_context": result.your_health_context,
        "next_steps": result.next_steps,
    }
