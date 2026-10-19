"""
LangGraph state schema for medical RAG agent.

Defines the state structure that flows through the graph nodes.
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class MedicalAgentState(TypedDict):
    """
    State schema for the medical RAG agent graph.

    Attributes:
        messages: Conversation history (managed by LangGraph's add_messages)
        ph_value: User's pH measurement
        health_profile: Dict containing age, symptoms, diagnoses, etc.
        display_query: User-perspective question for generation prompt (decoupled from retrieval query)
        docs_text: Formatted citation text from retrieval ("[1]: [Paper:page]: text")
        citations: List of citation metadata dicts
        used_citations: List of citation IDs actually used in the response
        overview: Short LLM-generated summary (from MedicalResponse.overview); passed to QueryResponse.overview
        your_ph: Deep dive 'Your pH' section content; None when not produced (e.g. no-profile branch)
        your_symptoms: Deep dive 'Your symptoms' section content; None when not produced (e.g. no-profile branch)
        your_personal_baseline: Deep dive 'Your personal baseline' section content; None when not produced (e.g. no-profile branch)
        your_health_context: Deep dive 'Your health context' section content; None when not produced (e.g. no-profile branch)
        next_steps: Deep dive 'Next steps' section content; None when not produced (e.g. no-profile branch)
        confidence_score: Confidence in retrieval quality (0.0-1.0)
        retrieval_quality: Assessment of retrieval quality ("high" or "low")
        confidence_method: How confidence was calculated ("score_only_high", "score_only_low", "hybrid")
        original_query: User's original query (saved by retrieve_node for refinement reference)
        retry_count: Number of retrieval retries attempted (0-2)
        refinement_history: List of refined queries attempted
        skip_retry: Whether to skip retry (True when refined query too similar to original)
    """

    messages: Annotated[list, add_messages]  # Auto-managed conversation history
    ph_value: float
    health_profile: dict[str, Any]
    display_query: str  # User-perspective question for generation prompt
    docs_text: str
    citations: list[dict[str, Any]]
    used_citations: list[int]  # Citation IDs used by LLM in response
    overview: str  # Short summary from MedicalResponse.overview; read by route layer for Overview tab
    your_ph: str | None  # Deep dive 'Your pH' section content
    your_symptoms: str | None  # Deep dive 'Your symptoms' section content
    your_personal_baseline: str | None  # Deep dive 'Your personal baseline' section content
    your_health_context: str | None  # Deep dive 'Your health context' section content
    next_steps: str | None  # Deep dive 'Next steps' section content
    confidence_score: float  # Confidence in retrieval quality (0.0-1.0)
    retrieval_quality: str  # "high" or "low"
    confidence_method: str  # "score_only_high", "score_only_low", or "hybrid"
    original_query: str  # User's original query (for refinement reference)
    retry_count: int  # Number of retrieval retries (0-2)
    refinement_history: list[str]  # Refined queries attempted
    skip_retry: bool  # True = skip retry (query too similar)
