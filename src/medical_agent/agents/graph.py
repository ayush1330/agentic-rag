"""
LangGraph workflow definition for medical RAG agent with agentic loop.

Builds agentic graph: retrieve → reasoning → [conditional] → refine_query ↔ retrieve (loop) → generate
Uses MemorySaver for multi-turn conversation memory.
Implements Phase 1 (metadata reranking), Phase 2 (confidence), and Phase 3 (agentic retry loop).
"""

import logging
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from medical_agent.agents.nodes import generate_node, retrieve_node
from medical_agent.agents.reasoning import reasoning_node
from medical_agent.agents.refine_query import refine_query_node
from medical_agent.agents.state import MedicalAgentState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_medical_rag_graph() -> "CompiledStateGraph":
    """
    Build and compile the medical RAG agent graph with agentic loop (Phase 3).

    Graph flow:
        START → retrieve → reasoning → [conditional]
                                       ├─ if confidence ≥ 0.7 OR retry_count ≥ 2 → generate
                                       └─ else → refine_query → [conditional]
                                                                 ├─ if skip_retry → generate
                                                                 └─ else → retrieve (LOOP)
                                          ↓
                                       END

    Features:
        - Phase 1: Metadata-weighted retrieval
        - Phase 2: Confidence assessment with conditional LLM validation
        - Phase 3: Agentic loop with query refinement + confidence-adaptive prompting
        - MemorySaver for multi-turn conversations
        - Conditional routing for retry logic
        - Max 2 retries (3 total retrieval attempts)

    Returns:
        Compiled LangGraph application with checkpointer
    """
    logger.info("Building medical RAG graph")

    # Create state graph
    workflow = StateGraph(MedicalAgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("refine_query", refine_query_node)  # Phase 3: Query refinement
    workflow.add_node("generate", generate_node)

    # Define edges with conditional routing (Phase 3 agentic loop)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "reasoning")

    # Conditional: From reasoning, decide to generate or refine query
    # If confidence >= 0.7 OR already retried twice, go to generate
    # Otherwise, go to refine_query for retry
    workflow.add_conditional_edges(
        "reasoning",
        lambda state: (
            "generate"
            if state.get("confidence_score", 0) >= 0.7 or state.get("retry_count", 0) >= 2
            else "refine_query"
        ),
    )

    # Conditional: From refine_query, decide to generate or loop back to retrieve
    # If skip_retry is True (query too similar/different), go to generate
    # Otherwise, go back to retrieve with refined query
    workflow.add_conditional_edges(
        "refine_query",
        lambda state: "generate" if state.get("skip_retry", True) else "retrieve",
    )

    # Final edge to END
    workflow.add_edge("generate", END)

    # Add memory for multi-turn conversations
    memory = MemorySaver()

    # Compile graph
    app = workflow.compile(checkpointer=memory)

    logger.info("Medical RAG graph compiled successfully")

    return app


# Global compiled graph instance
medical_rag_app = build_medical_rag_graph()
