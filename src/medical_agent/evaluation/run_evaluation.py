"""
RAGAS evaluation runner for the Medical RAG pipeline.

Runs test questions through the pipeline and evaluates responses
using RAGAS v0.4 metrics with LangSmith tracing.

Usage:
    python -m medical_agent.evaluation.run_evaluation --testset testsets/testset_XXXX.csv
"""

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from medical_agent.agents.graph import medical_rag_app
from medical_agent.evaluation.ragas_config import (
    get_evaluator_embeddings,
    get_evaluator_llm,
    setup_langsmith_tracing,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
# Keep defined for future golden-dataset evaluation, but do not use for now.
FACTUAL_CORRECTNESS_METRIC = FactualCorrectness(mode="f1")


def parse_docs_text_to_contexts(docs_text: str) -> list[str]:
    """
    Parse formatted citation text into a list of plain context strings.

    Converts "[1]: [Title:pN]: chunk text" format into plain text list
    suitable for RAGAS retrieved_contexts.

    Args:
        docs_text: Formatted citation text from the retrieval node.

    Returns:
        List of plain text context strings.
    """
    if not docs_text or docs_text == "No relevant medical research documents found.":
        return []

    contexts = []
    pattern = r"^\[\d+\]:\s*\[[^\]]+\]:\s*(.+)$"

    for block in docs_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        # Try to match the citation prefix pattern
        match = re.match(pattern, block, re.DOTALL)
        if match:
            contexts.append(match.group(1).strip())
        else:
            # Fallback: use the whole block if pattern doesn't match
            contexts.append(block)

    return contexts


async def run_pipeline_for_question(
    question: str,
    ph_value: float = 4.5,
    health_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single question through the medical RAG pipeline.

    Uses a unique thread_id per question to prevent MemorySaver
    from contaminating responses across test questions.

    Args:
        question: The test question to ask.
        ph_value: pH value for the health context.
        health_profile: Optional health profile dict.

    Returns:
        Dict with response, retrieved_contexts, citations, and elapsed_ms.
    """
    if health_profile is None:
        health_profile = {}

    thread_id = f"eval-{uuid.uuid4()}"

    start = time.perf_counter()

    result = await asyncio.to_thread(
        medical_rag_app.invoke,
        {
            "messages": [HumanMessage(content=question)],
            "ph_value": ph_value,
            "health_profile": health_profile,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Extract answer from last message
    answer = result["messages"][-1].content

    # Extract contexts from docs_text
    docs_text = result.get("docs_text", "")
    contexts = parse_docs_text_to_contexts(docs_text)

    return {
        "response": answer,
        "retrieved_contexts": contexts,
        "citations": result.get("citations", []),
        "elapsed_ms": round(elapsed_ms, 1),
    }


async def run_evaluation(
    testset_path: str | Path,
    metrics: list | None = None,
) -> Path:
    """
    Run RAGAS evaluation on a test set CSV.

    Args:
        testset_path: Path to the CSV test set (columns: user_input, reference).
        metrics: Optional list of RAGAS metrics. Defaults to standard set.

    Returns:
        Path to the saved JSON results file.
    """
    setup_langsmith_tracing("evaluation")

    testset_path = Path(testset_path)
    if not testset_path.exists():
        raise FileNotFoundError(f"Testset not found: {testset_path}")

    df = pd.read_csv(testset_path)
    required_cols = {"user_input", "reference"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Testset CSV missing required columns: {missing}")

    logger.info("Running evaluation on %d questions from %s", len(df), testset_path)

    evaluator_llm = get_evaluator_llm()
    evaluator_embeddings = get_evaluator_embeddings()

    # Run each question through the pipeline
    samples = []
    pipeline_results = []

    for idx, row in df.iterrows():
        question = row["user_input"]
        reference = row.get("reference", "")

        logger.info("[%d/%d] Processing: %s", idx + 1, len(df), question[:80])

        try:
            result = await run_pipeline_for_question(question)

            sample = SingleTurnSample(
                user_input=question,
                response=result["response"],
                retrieved_contexts=result["retrieved_contexts"],
                reference=str(reference) if pd.notna(reference) else "",
            )
            samples.append(sample)

            pipeline_results.append(
                {
                    "question": question,
                    "response": result["response"],
                    "num_contexts": len(result["retrieved_contexts"]),
                    "elapsed_ms": result["elapsed_ms"],
                    "status": "success",
                }
            )

        except Exception as e:
            logger.error("Failed on question %d: %s", idx + 1, e)
            pipeline_results.append(
                {
                    "question": question,
                    "status": "error",
                    "error": str(e),
                }
            )

    if not samples:
        raise RuntimeError("No successful pipeline runs. Cannot evaluate.")

    # Build evaluation dataset
    eval_dataset = EvaluationDataset(samples=samples)

    # Default metrics
    if metrics is None:
        metrics = [
            Faithfulness(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
            ResponseRelevancy(),
            # FACTUAL_CORRECTNESS_METRIC,  # Temporarily disabled
        ]

    logger.info(
        "Running RAGAS evaluation with %d metrics on %d samples", len(metrics), len(samples)
    )

    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"

    # Build output JSON
    per_sample_df = ragas_result.to_pandas()

    # Compute aggregate scores from per-sample DataFrame
    metric_names = [type(m).__name__ for m in metrics]
    aggregate_scores = {}
    for col in per_sample_df.columns:
        if col in ("user_input", "response", "retrieved_contexts", "reference"):
            continue
        values = pd.to_numeric(per_sample_df[col], errors="coerce").dropna()
        if not values.empty:
            aggregate_scores[col] = round(values.mean(), 4)

    output = {
        "metadata": {
            "testset": str(testset_path),
            "timestamp": timestamp,
            "num_questions": len(df),
            "num_successful": len(samples),
            "metrics": metric_names,
        },
        "aggregate_scores": aggregate_scores,
        "per_sample": per_sample_df.to_dict(orient="records"),
        "pipeline_results": pipeline_results,
    }

    output_path.write_text(json.dumps(output, indent=2, default=str))

    logger.info("Results saved to %s", output_path)
    print(f"\nEvaluation Results: {output_path}")
    print("Aggregate Scores:")
    for metric_name, score in aggregate_scores.items():
        print(f"  {metric_name}: {score}")

    return output_path


def main() -> None:
    """Run the evaluation CLI against a testset CSV and write JSON results.

    `--testset` is required and may be absolute or relative. Relative paths are
    resolved against the evaluation package directory when needed.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on medical RAG pipeline")
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help="Path to testset CSV (relative to evaluation/ or absolute)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve path relative to evaluation directory if not absolute
    testset_path = Path(args.testset)
    if not testset_path.is_absolute() and not testset_path.exists():
        testset_path = Path(__file__).parent / testset_path

    asyncio.run(run_evaluation(testset_path))


if __name__ == "__main__":
    main()
