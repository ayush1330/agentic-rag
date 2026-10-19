"""
RAGAS Evaluation Framework for Medical RAG Pipeline.

Provides synthetic test set generation and evaluation using RAGAS v0.4+
with Azure OpenAI and LangSmith tracing.
"""

from medical_agent.evaluation.generate_testset import generate_testset
from medical_agent.evaluation.ragas_config import (
    get_evaluator_embeddings,
    get_evaluator_llm,
)
from medical_agent.evaluation.run_evaluation import (
    parse_docs_text_to_contexts,
    run_evaluation,
    run_pipeline_for_question,
)

__all__ = [
    "get_evaluator_llm",
    "get_evaluator_embeddings",
    "generate_testset",
    "run_evaluation",
    "run_pipeline_for_question",
    "parse_docs_text_to_contexts",
]
