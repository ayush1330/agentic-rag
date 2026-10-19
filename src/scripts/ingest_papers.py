"""
Paper Ingestion Script - Interactive Mode

Downloads PDFs from GCP Cloud Storage and processes them through
the ingestion pipeline (parse → chunk → embed → store in pgvector).

Usage:
    python scripts/ingest_papers.py    # Interactive mode
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medical_agent.core.config import settings
from medical_agent.infrastructure.database.session import get_session_context, init_db
from medical_agent.infrastructure.gcp_storage import get_storage_client
from medical_agent.ingestion.pipeline import (
    MedicalIngestionPipeline,
    PipelineConfig,
    PipelineResult,
)
from medical_agent.ingestion.reference_sheet import (
    PaperReference,
    check_citation_coverage,
    load_reference_map,
    normalize_filename,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PART 0: NLM citation reference sheet location.
#
# Edit the absolute path below when the sheet is moved or renamed. Part 1 will
# replace this with a proper config mechanism. Keep as a module-level constant
# so every developer knows exactly where the file lives when running the
# script: `python3 scripts/ingest_papers.py`.
# -----------------------------------------------------------------------------
REFERENCE_SHEET_PATH = Path(
    "/Users/ayushsingh/Documents/pHera/AI Nation/RAG/Publications/"
    "pHera_publication_db_v2/20260607VM_RAG_v2_2_503_pdfs.xlsx"
)
# NOTE: absolute path to the v2 publication DB workbook. Update when moved/renamed.


class IngestionProgress:
    """Track progress of batch ingestion."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.succeeded = 0
        self.failed = 0
        self.results: list[PipelineResult] = []

    def update(self, result: PipelineResult) -> None:
        self.completed += 1
        self.results.append(result)
        if result.success:
            self.succeeded += 1
        else:
            self.failed += 1

    def print_progress(self, current_file: str) -> None:
        pct = (self.completed / self.total) * 100 if self.total > 0 else 0
        logger.info(
            f"[{self.completed}/{self.total}] ({pct:.0f}%) "
            f"Success: {self.succeeded}, Failed: {self.failed} | "
            f"Current: {current_file}"
        )

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Total papers processed: {self.completed}")
        print(f"Succeeded: {self.succeeded}")
        print(f"Failed: {self.failed}")
        print("-" * 60)

        if self.results:
            total_chunks = sum(r.chunk_count for r in self.results)
            total_embedded = sum(r.embedded_count for r in self.results)
            total_stored = sum(r.stored_count for r in self.results)
            total_time = sum(r.total_time_ms for r in self.results)

            print(f"Total chunks created: {total_chunks}")
            print(f"Total embeddings generated: {total_embedded}")
            print(f"Total chunks stored: {total_stored}")
            print(f"Total processing time: {total_time / 1000:.1f}s")

        if self.failed > 0:
            print("\nFailed papers:")
            for r in self.results:
                if not r.success:
                    print(f"  - {r.gcp_path}: {', '.join(r.errors)}")

        print("=" * 60)


async def check_azure_connectivity() -> bool:
    """Test Azure OpenAI connectivity."""
    logger.info("Testing Azure OpenAI connectivity...")
    try:
        from medical_agent.infrastructure.azure_openai import get_llama_index_embed_model

        embed_model = get_llama_index_embed_model()
        # Test with a simple embedding
        test_text = "test"
        embedding = await embed_model.aget_text_embedding(test_text)
        logger.info(f"✓ Azure OpenAI connectivity OK (embedding dim: {len(embedding)})")
        return True
    except Exception as e:
        logger.error(f"✗ Azure OpenAI connectivity failed: {e}")
        return False


def run_preflight_citation_check(
    pdf_files: list[str],
    ref_map: dict[str, PaperReference],
) -> None:
    """
    Hard pre-flight check: abort if any GCP paper is missing a citation.

    The team has explicitly asked for "proper ways to know every paper got
    a URL" — the cleanest implementation is to refuse to run at all when
    coverage is incomplete. The developer patches the sheet and re-runs.
    Partial ingestion is never allowed.
    """
    missing = check_citation_coverage(pdf_files, ref_map)
    if not missing:
        logger.info(
            "✓ Pre-flight citation check passed: all %d papers have a reference citation.",
            len(pdf_files),
        )
        return

    logger.error(
        "✗ Pre-flight citation check failed: %d of %d papers have no entry in the reference sheet.",
        len(missing),
        len(pdf_files),
    )
    print("\n" + "=" * 60, file=sys.stderr)
    print("MISSING REFERENCE CITATIONS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(
        f"The following {len(missing)} papers have no matching row in {REFERENCE_SHEET_PATH.name}.",
        file=sys.stderr,
    )
    print(
        "Patch column A of the 'Full DB_v2_503_pdfs' tab (must match the GCP "
        "filename exactly, case-insensitive) and re-run the script.",
        file=sys.stderr,
    )
    print("-" * 60, file=sys.stderr)
    for path in missing:
        print(f"  - {path}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    sys.exit(1)


async def ingest_papers(
    prefix: str = "papers/",
    limit: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> IngestionProgress:
    """
    Ingest papers from GCP bucket.

    Args:
        prefix: GCP path prefix to filter papers
        limit: Maximum number of papers to process
        dry_run: If True, only list papers without processing
        skip_existing: Skip papers already in the database

    Returns:
        IngestionProgress with results
    """
    # Verify configuration
    if not settings.is_gcp_configured():
        raise ValueError("GCP is not configured. Set GCP_PROJECT_ID and GCP_BUCKET_NAME in .env")

    if not settings.is_azure_openai_configured():
        raise ValueError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY and related vars in .env"
        )

    # Check Azure connectivity
    if not dry_run and not await check_azure_connectivity():
        raise ValueError("Azure OpenAI is not reachable. Check API key and endpoint.")

    # Initialize database
    logger.info("Initializing database...")
    await init_db()

    # Get storage client
    storage = get_storage_client()

    # Verify GCP connection
    logger.info(f"Connecting to GCP bucket: {settings.gcp_bucket_name}")
    storage.verify_connection()

    # List PDFs
    logger.info(f"Listing PDFs with prefix: {prefix}")
    pdf_files = storage.list_pdfs(prefix=prefix)

    if limit:
        pdf_files = pdf_files[:limit]

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    if not pdf_files:
        logger.warning("No PDF files found!")
        return IngestionProgress(0)

    # Part 0: load the NLM citation reference sheet and validate coverage.
    ref_map = load_reference_map(REFERENCE_SHEET_PATH)
    run_preflight_citation_check(pdf_files, ref_map)

    # Preview mode
    if dry_run:
        print("\nDRY RUN - Papers to be processed:")
        print("-" * 60)
        for i, pdf_name in enumerate(pdf_files, 1):
            print(f"{i}. {pdf_name}")
        print("-" * 60)
        print(f"Total: {len(pdf_files)} papers")
        return IngestionProgress(len(pdf_files))

    # Initialize pipeline
    pipeline = MedicalIngestionPipeline(
        config=PipelineConfig(
            max_chunk_tokens=512,
            extract_metadata=True,
        )
    )

    progress = IngestionProgress(len(pdf_files))

    # Process each paper
    for gcp_path in pdf_files:
        progress.print_progress(gcp_path)

        try:
            # Download PDF
            logger.info(f"[{progress.completed + 1}/{progress.total}] Downloading: {gcp_path}")
            pdf_content = storage.download_pdf(gcp_path)
            logger.info(f"Downloaded {len(pdf_content)} bytes")

            ref_key = normalize_filename(Path(gcp_path).stem)
            paper_reference = ref_map[ref_key]

            # Process through pipeline
            logger.info(
                f"[{progress.completed + 1}/{progress.total}] Starting pipeline for: {gcp_path}"
            )
            async with get_session_context() as session:
                result = await pipeline.process_paper(
                    session=session,
                    pdf_content=pdf_content,
                    gcp_path=f"gs://{settings.gcp_bucket_name}/{gcp_path}",
                    skip_duplicate_check=not skip_existing,
                    paper_reference=paper_reference,
                )
                # Context manager commits on success, rollback handled in pipeline

            progress.update(result)

            if result.success:
                logger.info(
                    f"SUCCESS: {result.paper_title or gcp_path} - "
                    f"{result.stored_count} chunks stored in {result.total_time_ms}ms"
                )
            else:
                logger.warning(f"SKIPPED/FAILED: {gcp_path} - {result.errors}")

        except Exception as e:
            logger.error(f"ERROR processing {gcp_path}: {e}")
            # Create a failed result
            from uuid import uuid4

            failed_result = PipelineResult(
                paper_id=uuid4(),
                paper_title=None,
                gcp_path=gcp_path,
                errors=[str(e)],
            )
            progress.update(failed_result)

    progress.print_summary()
    return progress


def display_papers_list(papers: list[str]) -> None:
    """Display papers with numbered list."""
    print("\n" + "-" * 80)
    print(f"Available Papers ({len(papers)} total):")
    print("-" * 80)
    for i, pdf_name in enumerate(papers, 1):
        print(f"{i}. {pdf_name}")
    print("-" * 80)


def parse_selected_ids(input_str: str, max_index: int) -> list[int]:
    """Parse comma-separated numbers from user input."""
    try:
        indices = [int(x.strip()) - 1 for x in input_str.split(",")]
        # Validate indices
        if any(i < 0 or i >= max_index for i in indices):
            raise ValueError("Index out of range")
        return indices
    except (ValueError, IndexError):
        return []


async def fetch_ingested_gcp_paths() -> set[str]:
    """Return the gcp_path values already present in the papers table.

    Used to hide already-ingested papers from the interactive menu so it only
    offers not-yet-ingested work. Matches the same key the pipeline's duplicate
    check uses (Paper.gcp_path == "gs://{bucket}/{blob_name}").
    """
    from sqlalchemy import select

    from medical_agent.infrastructure.database.models import Paper

    async with get_session_context() as session:
        rows = await session.execute(select(Paper.gcp_path))
        return {path for (path,) in rows.all() if path}


async def interactive_mode() -> None:
    """Interactive mode for ingesting papers."""
    print("\n" + "=" * 80)
    print("PAPER INGESTION - INTERACTIVE MODE")
    print("=" * 80)

    # Verify configuration
    if not settings.is_gcp_configured():
        print("ERROR: GCP is not configured. Set GCP_PROJECT_ID and GCP_BUCKET_NAME in .env")
        return

    if not settings.is_azure_openai_configured():
        print(
            "ERROR: Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY and related vars in .env"
        )
        return

    # Get storage client and list all papers
    storage = get_storage_client()
    try:
        logger.info(f"Connecting to GCP bucket: {settings.gcp_bucket_name}")
        storage.verify_connection()
    except Exception as e:
        print(f"ERROR: Failed to connect to GCP bucket: {e}")
        return

    logger.info("Listing all available PDFs...")
    all_pdfs = storage.list_pdfs(prefix="")

    if not all_pdfs:
        print("No PDF files found in bucket!")
        return

    # Hide already-ingested papers so the menu only offers not-yet-ingested work.
    # Falls back to the full bucket list if the DB can't be reached.
    try:
        ingested = await fetch_ingested_gcp_paths()
        bucket = settings.gcp_bucket_name
        remaining = [p for p in all_pdfs if f"gs://{bucket}/{p}" not in ingested]
    except Exception as e:
        logger.warning(
            f"Could not read ingested papers from DB ({e}); showing full bucket list"
        )
        remaining = all_pdfs

    already = len(all_pdfs) - len(remaining)
    print(
        f"Found {len(all_pdfs)} in bucket · {already} already ingested · "
        f"{len(remaining)} remaining"
    )

    if not remaining:
        print("\nAll papers in the bucket are already ingested. Nothing to do.")
        return

    while True:
        print("\nOptions:")
        print("1. List papers")
        print("2. Ingest all papers")
        print("3. Ingest by selecting IDs")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            # List not-yet-ingested papers
            display_papers_list(remaining)

        elif choice == "2":
            # Ingest all not-yet-ingested papers
            display_papers_list(remaining)
            confirm = input(f"\nIngest all {len(remaining)} papers? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Ingestion cancelled.")
                continue

            # Check Azure connectivity
            if not await check_azure_connectivity():
                print("ERROR: Azure OpenAI is not reachable. Check API key and endpoint.")
                continue

            # Initialize database
            logger.info("Initializing database...")
            await init_db()

            # Perform ingestion
            print("\nStarting ingestion...")
            await ingest_papers(
                prefix="",
                limit=None,
                dry_run=False,
                skip_existing=True,
            )

        elif choice == "3":
            # Ingest by selecting IDs (from not-yet-ingested papers)
            display_papers_list(remaining)

            ids_input = input("\nEnter paper IDs to ingest (e.g., 1,3,5,8): ").strip()
            selected_indices = parse_selected_ids(ids_input, len(remaining))

            if not selected_indices:
                print("Invalid input. Please enter valid paper numbers.")
                continue

            selected_papers = [remaining[i] for i in selected_indices]

            # Show preview
            print("\n" + "-" * 80)
            print(f"Papers to ingest ({len(selected_papers)} selected):")
            print("-" * 80)
            for paper in selected_papers:
                print(f"  • {paper}")
            print("-" * 80)

            confirm = input(f"\nIngest {len(selected_papers)} papers? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Ingestion cancelled.")
                continue

            # Check Azure connectivity
            if not await check_azure_connectivity():
                print("ERROR: Azure OpenAI is not reachable. Check API key and endpoint.")
                continue

            # Initialize database
            logger.info("Initializing database...")
            await init_db()

            # Part 0: ensure selected papers all have a reference citation.
            ref_map = load_reference_map(REFERENCE_SHEET_PATH)
            run_preflight_citation_check(selected_papers, ref_map)

            # Perform ingestion for selected papers
            print("\nStarting ingestion...")
            pipeline = MedicalIngestionPipeline(
                config=PipelineConfig(
                    max_chunk_tokens=512,
                    extract_metadata=True,
                )
            )

            progress = IngestionProgress(len(selected_papers))

            for gcp_path in selected_papers:
                progress.print_progress(gcp_path)

                try:
                    logger.info(
                        f"[{progress.completed + 1}/{progress.total}] Downloading: {gcp_path}"
                    )
                    pdf_content = storage.download_pdf(gcp_path)
                    logger.info(f"Downloaded {len(pdf_content)} bytes")

                    ref_key = normalize_filename(Path(gcp_path).stem)
                    paper_reference = ref_map[ref_key]

                    logger.info(
                        f"[{progress.completed + 1}/{progress.total}] Starting pipeline for: {gcp_path}"
                    )
                    async with get_session_context() as session:
                        result = await pipeline.process_paper(
                            session=session,
                            pdf_content=pdf_content,
                            gcp_path=f"gs://{settings.gcp_bucket_name}/{gcp_path}",
                            skip_duplicate_check=True,
                            paper_reference=paper_reference,
                        )
                        # Context manager commits on success, rollback handled in pipeline

                    progress.update(result)

                    if result.success:
                        logger.info(
                            f"SUCCESS: {result.paper_title or gcp_path} - "
                            f"{result.stored_count} chunks stored in {result.total_time_ms}ms"
                        )
                    else:
                        logger.warning(f"SKIPPED/FAILED: {gcp_path} - {result.errors}")

                except Exception as e:
                    logger.error(f"ERROR processing {gcp_path}: {e}")
                    from uuid import uuid4

                    failed_result = PipelineResult(
                        paper_id=uuid4(),
                        paper_title=None,
                        gcp_path=gcp_path,
                        errors=[str(e)],
                    )
                    progress.update(failed_result)

            progress.print_summary()

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


def main() -> None:
    try:
        asyncio.run(interactive_mode())
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
