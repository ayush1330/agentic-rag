"""
Synthetic test set generation from ingested paper chunks.

Fetches chunks from the database and uses RAGAS TestsetGenerator
to create synthetic question-answer pairs for evaluation.

Usage:
    python -m medical_agent.evaluation.generate_testset --size 20
    python -m medical_agent.evaluation.generate_testset --size 20 --seed 42  # reproducible
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import text

from medical_agent.evaluation.ragas_config import (
    get_evaluator_embeddings,
    get_evaluator_llm,
    setup_langsmith_tracing,
)
from medical_agent.infrastructure.database.session import get_session_context

logger = logging.getLogger(__name__)

TESTSETS_DIR = Path(__file__).parent / "testsets"

# Instructions for RAGAS to generate questions from end-user POV
MEDICAL_LLM_CONTEXT = """Most questions should be from the point of view of a user who has provided their health profile and vaginal pH value, and wants personalised answers about what the research means for her. She may ask follow-up questions about technical medical terms. For evaluation diversity, some generic or non-personalised questions are acceptable."""


async def fetch_chunks_as_langchain_docs(
    limit: int = 200,
    paper_title_filter: str | None = None,
    seed: int | None = None,
) -> list[LangchainDocument]:
    """
    Fetch paper chunks from the database as LangChain Documents.

    Uses raw SQL to avoid loading 3072-dim embedding vectors into memory.
    Orders by RANDOM() for diversity across papers; pass seed for reproducibility.

    Args:
        limit: Maximum number of chunks to fetch.
        paper_title_filter: Optional filter to only fetch chunks from a specific paper.
        seed: Optional seed for reproducible random ordering. If None, uses RANDOM().

    Returns:
        List of LangchainDocument objects with page_content and metadata.
    """
    query = "SELECT text, metadata_, node_id FROM data_paper_chunks"
    params: dict = {}

    if paper_title_filter:
        query += " WHERE metadata_->>'title' ILIKE :title"
        params["title"] = f"%{paper_title_filter}%"

    if seed is not None:
        query += " ORDER BY md5(COALESCE(node_id::text, '') || :seed)"
        params["seed"] = str(seed)
    else:
        query += " ORDER BY RANDOM()"

    query += " LIMIT :limit"
    params["limit"] = limit

    documents = []
    async with get_session_context() as session:
        result = await session.execute(text(query), params)
        rows = result.fetchall()

        for row in rows:
            chunk_text = row[0]
            metadata = row[1] if row[1] else {}
            node_id = row[2]

            if not chunk_text or not chunk_text.strip():
                continue

            # Build clean metadata for RAGAS
            doc_metadata = {
                "node_id": node_id,
                "title": metadata.get("title", "Unknown"),
            }
            if "doc_items" in metadata:
                doc_items = metadata["doc_items"]
                if doc_items and doc_items[0].get("prov"):
                    doc_metadata["page"] = doc_items[0]["prov"][0].get("page_no", "N/A")

            documents.append(
                LangchainDocument(
                    page_content=chunk_text.strip(),
                    metadata=doc_metadata,
                )
            )

    logger.info("Fetched %d chunks from database", len(documents))
    return documents


def _build_query_distribution(
    llm: Any,
    llm_context: str | None = None,
) -> list[tuple[Any, float]]:
    """Build weighted RAGAS query synthesizer distribution for test generation.

    Returns an 80/20 split between single-hop and multi-hop styles:
    - 80% single-hop specific questions
    - 10% multi-hop abstract questions
    - 10% multi-hop specific questions
    """
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    return [
        (SingleHopSpecificQuerySynthesizer(llm=llm, llm_context=llm_context), 0.8),
        (MultiHopAbstractQuerySynthesizer(llm=llm, llm_context=llm_context), 0.1),
        (MultiHopSpecificQuerySynthesizer(llm=llm, llm_context=llm_context), 0.1),
    ]


async def generate_testset(
    testset_size: int = 20,
    limit_chunks: int = 200,
    paper_title_filter: str | None = None,
    seed: int | None = None,
) -> Path:
    """
    Generate a synthetic test set from database chunks using RAGAS.

    Args:
        testset_size: Number of test questions to generate.
        limit_chunks: Maximum chunks to fetch from the database.
        paper_title_filter: Optional filter for specific paper title.
        seed: Optional seed for reproducible chunk ordering.

    Returns:
        Path to the saved CSV test set file.
    """
    from ragas.testset import TestsetGenerator

    setup_langsmith_tracing("testset-generation")

    logger.info(
        "Generating testset: size=%d, chunk_limit=%d, filter=%s, seed=%s",
        testset_size,
        limit_chunks,
        paper_title_filter,
        seed,
    )

    documents = await fetch_chunks_as_langchain_docs(limit_chunks, paper_title_filter, seed=seed)
    if not documents:
        raise ValueError("No chunks found in database. Run paper ingestion first.")

    llm = get_evaluator_llm()
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=get_evaluator_embeddings(),
        llm_context=MEDICAL_LLM_CONTEXT,
    )

    query_distribution = _build_query_distribution(llm, llm_context=MEDICAL_LLM_CONTEXT)

    # Use generate_with_chunks: our DB chunks are pre-chunked, so we skip
    # HeadlinesExtractor/HeadlineSplitter (which cause "headlines not found" errors
    # when documents are short or already chunked)
    testset = generator.generate_with_chunks(
        chunks=documents,
        testset_size=testset_size,
        query_distribution=query_distribution,
    )

    # Save as timestamped CSV
    TESTSETS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = TESTSETS_DIR / f"testset_{timestamp}.csv"

    df = testset.to_pandas()
    df.to_csv(output_path, index=False)

    logger.info("Testset saved to %s (%d rows)", output_path, len(df))
    print(f"Testset saved: {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")

    return output_path


def main() -> None:
    """Run the testset-generation CLI and persist a timestamped CSV artifact.

    Required input is provided via flags (no positional args). `--size` and
    `--limit` control generation volume, while `--paper` and `--seed` are
    optional filters for targeted and reproducible runs.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate RAGAS synthetic test set")
    parser.add_argument("--size", type=int, default=20, help="Number of test questions")
    parser.add_argument("--limit", type=int, default=200, help="Max chunks to fetch")
    parser.add_argument("--paper", type=str, default=None, help="Filter by paper title")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible chunk ordering (enables diverse paper coverage)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(
        generate_testset(
            testset_size=args.size,
            limit_chunks=args.limit,
            paper_title_filter=args.paper,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
