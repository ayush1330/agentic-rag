"""
Medical RAG using LangGraph + LlamaIndex.

Phase 2: Post-retrieval RAG workflow with MemorySaver for multi-turn conversations.

Usage:
    from medical_agent.agents import medical_rag_app

    # Run the graph
    result = medical_rag_app.invoke(
        {
            "messages": [{"role": "user", "content": "What causes high pH?"}],
            "ph_value": 5.2,
            "health_profile": {"age": 28, "symptoms": {"discharge": ["Creamy"]}}
        },
        config={"configurable": {"thread_id": "session_123"}}
    )
"""

from medical_agent.agents.graph import medical_rag_app
from medical_agent.agents.llamaindex_retrieval import (
    build_retriever,
    retrieve_nodes,
)

__all__ = [
    "medical_rag_app",
    "build_retriever",
    "retrieve_nodes",
]
