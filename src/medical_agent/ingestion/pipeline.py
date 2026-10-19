"""
LlamaIndex-Based Ingestion Pipeline (Phase 2: Enhanced Data Quality)

Modern ingestion pipeline using LlamaIndex components + custom quality filters:
1. Parse PDFs with DoclingReader (JSON export for structured tables AND sections)
2. Chunk with DoclingNodeParser + HybridChunker (section-aware + token-limited, 1024 tokens)
3. **Filter chunks** (remove bibliography, headers, noise - removes 40-60% garbage)
4. **Transform tables** (table-to-natural-language with gpt-4o)
5. **Add contextual headers** (Anthropic's approach - 49% error reduction)
6. Extract medical metadata (8 fields)
7. Generate embeddings (Azure OpenAI 3072-dim)
8. Store in pgvector

Phase 2 improvements target data quality at ingestion time to improve retrieval precision.
"""

import hashlib
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

from docling.chunking import HybridChunker
from llama_index.core.schema import Document
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.core.config import settings
from medical_agent.infrastructure.database.models import Paper
from medical_agent.infrastructure.database.session import get_session_context
from medical_agent.ingestion.chunk_filter import filter_chunks
from medical_agent.ingestion.contextual_chunking import add_contextual_headers
from medical_agent.ingestion.metadata import extract_medical_metadata, stamp_metadata_on_nodes
from medical_agent.ingestion.reference_sheet import PaperReference, normalize_filename
from medical_agent.ingestion.table_transformer import transform_table_chunks

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the LlamaIndex ingestion pipeline."""

    # Docling reader settings
    docling_export_type: str = "json"  # Use JSON for structured tables AND section metadata
    docling_enable_ocr: bool = True  # Auto OCR for scanned papers
    docling_table_structure_mode: str = "accurate"  # TableFormer accurate mode

    # Metadata extraction settings
    extract_metadata: bool = True

    # Chunking settings for HybridChunker
    max_chunk_tokens: int = 1024  # Maximum tokens per chunk (optimized for 3072-dim embeddings)
    min_chunk_tokens: int = 64  # Minimum tokens per chunk (avoids tiny chunks)
    merge_small_chunks: bool = True  # Merge small adjacent chunks

    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072


@dataclass
class PipelineResult:
    """Result of a complete LlamaIndex pipeline run."""

    paper_id: uuid.UUID
    paper_title: str | None
    gcp_path: str

    # Pipeline stages
    parsed: bool = False
    chunked: bool = False
    embedded: bool = False
    stored: bool = False

    # Statistics
    chunk_count: int = 0
    node_count: int = 0
    embedded_count: int = 0  # For backward compatibility with script
    stored_count: int = 0
    failed_count: int = 0

    # Timing
    parse_time_ms: int = 0
    pipeline_time_ms: int = 0
    store_time_ms: int = 0
    total_time_ms: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.stored and self.failed_count == 0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"LlamaIndex Pipeline Result: {status}",
            f"  Paper: {self.paper_title or 'Unknown'} ({self.paper_id})",
            f"  GCP Path: {self.gcp_path}",
            f"  Nodes: {self.node_count} → Stored: {self.stored_count}",
            f"  Failed: {self.failed_count}",
            f"  Time: {self.total_time_ms}ms (parse: {self.parse_time_ms}, pipeline: {self.pipeline_time_ms}, store: {self.store_time_ms})",
        ]
        if self.errors:
            lines.append(f"  Errors: {', '.join(self.errors)}")
        return "\n".join(lines)


class MedicalIngestionPipeline:
    """
    LlamaIndex-based ingestion pipeline for medical papers (Phase 2: Enhanced).

    Pipeline stages:
    1. Parse PDF with DoclingReader (JSON export for structured tables)
    2. Chunk with DoclingNodeParser + HybridChunker (1024 tokens, section-aware)
    3. **Filter chunks** (remove structural garbage: bibliography, headers, noise)
    4. **Transform tables** (convert to natural language with gpt-4o)
    5. **Add contextual headers** (Anthropic's approach for self-contained chunks)
    6. Extract medical metadata (8 fields)
    7. Generate embeddings (Azure OpenAI 3072-dim)
    8. Store in pgvector (native PGVectorStore with hybrid_search=True)

    Phase 2 improvements target data quality at ingestion to improve retrieval metrics.

    Args:
        config: Pipeline configuration
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()

        # Initialize native PGVectorStore with hybrid search support.
        # Connection strings use @localhost:5432?host= pattern so SQLAlchemy
        # URL parsing succeeds while the actual socket/host is overridden at
        # driver level via the ?host= query param (works for both Cloud SQL
        # Unix sockets and regular TCP hosts).
        host = settings.resolved_postgres_host
        port = settings.resolved_postgres_port
        user = settings.resolved_postgres_user
        password = settings.resolved_postgres_password
        db = settings.resolved_postgres_db
        encoded_password = quote_plus(password)

        if host.startswith("/"):
            sync_conn_str = (
                f"postgresql+psycopg2://{user}:{encoded_password}@localhost:5432/{db}?host={host}"
            )
            async_conn_str = (
                f"postgresql+asyncpg://{user}:{encoded_password}@localhost:5432/{db}?host={host}"
            )
        else:
            sync_conn_str = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{db}"
            async_conn_str = f"postgresql+asyncpg://{user}:{encoded_password}@{host}:{port}/{db}"

        # Table name: LlamaIndex prefixes with "data_", so "paper_chunks" → "data_paper_chunks"
        self.vector_store = PGVectorStore.from_params(
            connection_string=sync_conn_str,
            async_connection_string=async_conn_str,
            table_name="paper_chunks",  # Will use data_paper_chunks (LlamaIndex adds "data_" prefix)
            embed_dim=settings.embedding_dimension,
            hybrid_search=True,  # Enable built-in BM25 + vector fusion
            text_search_config="english",  # Full-text search language
            perform_setup=False,  # Use existing table schema (don't create new table)
        )

        # Initialize Docling reader with optimal settings
        self.reader = DoclingReader(
            export_type=self.config.docling_export_type,
            ocr=self.config.docling_enable_ocr,
            table_structure_mode=self.config.docling_table_structure_mode,
        )

        # Initialize embedding model
        # Determine which credentials to use
        if settings.azure_openai_embedding_api_key and settings.azure_openai_embedding_endpoint:
            # Use separate embedding credentials
            api_key = settings.azure_openai_embedding_api_key
            azure_endpoint = settings.azure_openai_embedding_endpoint
            api_version = settings.azure_openai_embedding_api_version
            deployment_name = settings.azure_openai_embedding_deployment_name
        else:
            # Fall back to main credentials
            api_key = settings.azure_openai_api_key
            azure_endpoint = settings.azure_openai_endpoint
            api_version = settings.azure_openai_api_version
            deployment_name = settings.azure_openai_embedding_deployment_name

        self.embedding_model = AzureOpenAIEmbedding(
            model=self.config.embedding_model,
            deployment_name=deployment_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            dimensions=self.config.embedding_dimensions,
        )

        # Initialize HybridChunker for token-aware section chunking
        # Use default tokenizer (HybridChunker has its own tokenizer handling)
        hybrid_chunker = HybridChunker(
            max_tokens=self.config.max_chunk_tokens,  # 1024 tokens: optimized for 3072-dim embeddings
            min_tokens=self.config.min_chunk_tokens,  # Avoid tiny chunks
            merge_peers=self.config.merge_small_chunks,  # Merge small adjacent chunks
            heading_as_metadata=True,  # Preserve section hierarchy
        )

        # Initialize DoclingNodeParser with HybridChunker
        # Splits Docling JSON by structure + enforces token limits
        self.node_parser = DoclingNodeParser(chunker=hybrid_chunker)

        # Note: Pipeline runs custom processing steps (filter, transform, context)
        # before metadata extraction and embedding. See process_paper() for flow.

        logger.info("Initialized LlamaIndex ingestion pipeline (Phase 2: Enhanced)")
        logger.info(
            "Pipeline flow: Parse → Chunk(1024) → Metadata(doc-level) → Filter → Table-to-NL → Context → Stamp → Embed"
        )

    def compute_file_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content for deduplication."""
        return hashlib.sha256(content).hexdigest()

    async def process_paper(
        self,
        session: AsyncSession,
        pdf_content: bytes,
        gcp_path: str,
        paper_id: uuid.UUID | None = None,
        skip_duplicate_check: bool = False,
        paper_reference: PaperReference | None = None,
    ) -> PipelineResult:
        """
        Process a PDF through the LlamaIndex ingestion pipeline.

        Args:
            session: Database session
            pdf_content: Raw PDF bytes
            gcp_path: GCP Cloud Storage path to the PDF
            paper_id: Optional paper ID (defaults to new UUID)
            skip_duplicate_check: If False, check for existing paper by hash/path

        Returns:
            PipelineResult with processing details
        """
        start_time = time.monotonic()

        # Initialize result
        result = PipelineResult(
            paper_id=paper_id or uuid.uuid4(),
            paper_title=None,
            gcp_path=gcp_path,
        )

        file_hash = self.compute_file_hash(pdf_content)

        # Check for duplicates unless skipped
        if not skip_duplicate_check:
            existing = await session.execute(
                select(Paper).where((Paper.file_hash == file_hash) | (Paper.gcp_path == gcp_path))
            )
            existing_paper = existing.scalar_one_or_none()
            if existing_paper:
                result.errors.append(
                    f"Duplicate paper: already exists as '{existing_paper.title}' (id: {existing_paper.id})"
                )
                result.total_time_ms = int((time.monotonic() - start_time) * 1000)
                logger.warning(f"Skipping duplicate paper: {gcp_path}")
                return result

        try:
            # Stage 1: Parse PDF with DoclingReader
            parse_start = time.monotonic()
            try:
                # DoclingReader expects a file path, so write to temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                    tmp_file.write(pdf_content)
                    tmp_path = tmp_file.name

                try:
                    logger.info(f"Starting PDF parse for {gcp_path} ({len(pdf_content)} bytes)")
                    documents = list(self.reader.lazy_load_data(file_path=tmp_path))
                    logger.info(f"PDF parse completed: extracted {len(documents)} documents")
                    result.parsed = True
                finally:
                    # Clean up temp file
                    Path(tmp_path).unlink(missing_ok=True)

                if not documents:
                    result.errors.append("No documents extracted from PDF")
                    return result

                # Get the first document (DoclingReader typically returns one document per PDF)
                document = documents[0]
                result.paper_title = (
                    document.metadata.get("title") or Path(gcp_path).stem or "Unknown"
                )

                # Add paper metadata to document
                # NOTE: paper_id is NOT stamped here — the canonical ID is only known
                # after _upsert_paper() resolves the DB row (see below).
                document.metadata["gcp_path"] = gcp_path
                document.id_ = str(result.paper_id)  # Provisional; nodes get canonical ID post-upsert

            except Exception as e:
                logger.error(f"PDF parsing failed: {e}", exc_info=True)
                result.errors.append(f"Parse error: {e}")
                return result
            finally:
                result.parse_time_ms = int((time.monotonic() - parse_start) * 1000)

            # Stage 2: Chunk and process
            pipeline_start = time.monotonic()
            try:
                logger.info(
                    f"Starting pipeline (chunk + filter + transform + metadata + embed) for paper {result.paper_id}"
                )

                # Step 2.1: Chunk with DoclingNodeParser
                logger.info("Step 1/6: Chunking document...")
                nodes = self.node_parser.get_nodes_from_documents([document])
                logger.info(f"✓ Chunked: {len(nodes)} raw chunks")
                result.chunked = True

                # Step 2.2: Extract metadata from RAW chunks (before filtering)
                # Uses title + abstract/intro for unique per-paper metadata
                # Keep this before filtering/context transforms so extraction sees
                # the highest-fidelity source text from the original parse.
                logger.info("Step 2/7: Extracting medical metadata from document...")
                medical_metadata = await extract_medical_metadata(
                    title=result.paper_title, nodes=nodes
                )
                logger.info(
                    f"✓ Metadata extracted (confidence: {medical_metadata.get('confidence', 0.0):.2f})"
                )

                # Step 2.3: Filter out structural garbage
                logger.info("Step 3/7: Filtering structural noise...")
                nodes = filter_chunks(nodes)
                logger.info(f"✓ Filtered: {len(nodes)} chunks after filter")

                # Capture raw_text BEFORE any text mutations (table transform, contextual headers)
                raw_text_snapshot: dict[str, str] = {
                    node.node_id: node.get_content() for node in nodes
                }
                logger.info(f"Captured raw_text for {len(raw_text_snapshot)} nodes")

                # Step 2.4: Transform tables to natural language
                logger.info("Step 4/7: Transforming tables to natural language...")
                nodes = await transform_table_chunks(nodes)
                logger.info(f"✓ Tables transformed: {len(nodes)} chunks")

                # Step 2.5: Add contextual headers (Anthropic's approach)
                logger.info("Step 5/7: Adding contextual headers...")
                nodes = await add_contextual_headers(nodes, title=result.paper_title)
                logger.info(f"✓ Context added: {len(nodes)} chunks")

                # Step 2.6: Stamp metadata on processed chunks
                # Stamp after all structural transforms so every persisted node
                # carries a consistent, final metadata payload.
                logger.info("Step 6/7: Stamping metadata on chunks...")
                nodes = stamp_metadata_on_nodes(nodes, medical_metadata)
                logger.info(f"✓ Metadata stamped on {len(nodes)} chunks")


                # Step 2.8: Generate embeddings
                logger.info("Step 7/7: Generating embeddings...")
                nodes = list(await self.embedding_model.acall(nodes))
                logger.info(f"✓ Pipeline completed: {len(nodes)} embedded nodes")

                result.embedded = True
                result.node_count = len(nodes)
                result.chunk_count = len(nodes)
                result.embedded_count = len(nodes)

            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}", exc_info=True)
                result.errors.append(f"Pipeline error: {e}")
                return result
            finally:
                result.pipeline_time_ms = int((time.monotonic() - pipeline_start) * 1000)

            if not nodes:
                result.errors.append("No nodes generated from pipeline")
                return result

            # Stage 5: Store in database
            store_start = time.monotonic()
            try:
                # Create paper record (title comes from Docling via document.metadata)
                logger.info(f"Creating paper record for {result.paper_id}")
                paper = await self._upsert_paper(
                    session=session,
                    paper_id=result.paper_id,
                    gcp_path=gcp_path,
                    file_hash=file_hash,
                    paper_reference=paper_reference,
                )
                result.paper_id = paper.id
                result.paper_title = paper.title
                logger.info(f"Paper record created: {paper.title}")

                # Stamp canonical paper_id on all nodes now that the DB row is resolved.
                # On re-ingestion the upsert returns the existing row's UUID, which may
                # differ from the provisional UUID minted at pipeline start. Stamping
                # here ensures every chunk carries the correct foreign key.
                canonical_paper_id = str(paper.id)
                for node in nodes:
                    if node.metadata is None:
                        node.metadata = {}
                    node.metadata["paper_id"] = canonical_paper_id
                logger.info(f"Stamped canonical paper_id on {len(nodes)} nodes")

                # Store nodes using native PGVectorStore
                logger.info(f"Starting to store {len(nodes)} nodes to vector store")
                node_ids = self.vector_store.add(nodes)
                logger.info(f"Successfully stored {len(node_ids)} nodes")
                result.stored = True
                result.stored_count = len(node_ids)

                # Persist raw_text via AsyncSession (separate from vector_store's psycopg2 connection)
                raw_text_updated = 0
                for node_id_str in node_ids:
                    if node_id_str in raw_text_snapshot:
                        await session.execute(
                            text(
                                "UPDATE data_paper_chunks SET raw_text = :r WHERE node_id = :n"
                            ),
                            {"r": raw_text_snapshot[node_id_str], "n": node_id_str},
                        )
                        raw_text_updated += 1
                logger.info(f"raw_text persisted on {raw_text_updated}/{len(node_ids)} chunks")

                # Mark paper as processed
                paper.is_processed = True
                await session.flush()

            except Exception as e:
                logger.error(f"Storage failed: {e}", exc_info=True)
                result.errors.append(f"Store error: {e}")
                # Keep storage atomic: undo the inserted Paper row if vector-store
                # add fails, so we do not leave a "processed" paper without chunks.
                await session.rollback()
                return result
            finally:
                result.store_time_ms = int((time.monotonic() - store_start) * 1000)

        finally:
            result.total_time_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(result.summary())
        return result

    async def _upsert_paper(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        gcp_path: str,
        file_hash: str,
        paper_reference: PaperReference | None,
    ) -> Paper:
        """Upsert paper row; ON CONFLICT (gcp_file_name) DO UPDATE SET."""
        gcp_file_name = normalize_filename(Path(gcp_path).stem)

        # Derive title: prefer reference sheet, fall back to filename stem
        title = (paper_reference.title if paper_reference and paper_reference.title
                 else Path(gcp_path).stem or "Untitled")

        values: dict = {
            "id": paper_id,
            "gcp_file_name": gcp_file_name,
            "title": title,
            "gcp_path": gcp_path,
            "file_hash": file_hash,
            "is_processed": False,
            "nlm_reference": paper_reference.nlm_reference if paper_reference else None,
            "pmid": paper_reference.pmid if paper_reference else None,
            "pmcid": paper_reference.pmcid if paper_reference else None,
            "doi": paper_reference.doi if paper_reference else None,
        }

        stmt = pg_insert(Paper).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["gcp_file_name"],
            set_={
                "title": stmt.excluded.title,
                "gcp_path": stmt.excluded.gcp_path,
                "file_hash": stmt.excluded.file_hash,
                "nlm_reference": stmt.excluded.nlm_reference,
                "pmid": stmt.excluded.pmid,
                "pmcid": stmt.excluded.pmcid,
                "doi": stmt.excluded.doi,
            },
        )
        await session.execute(stmt)
        await session.flush()

        result = await session.execute(select(Paper).where(Paper.gcp_file_name == gcp_file_name))
        return result.scalar_one()


# Convenience functions


async def process_pdf_llamaindex(
    pdf_content: bytes,
    gcp_path: str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Process a PDF through the LlamaIndex ingestion pipeline.

    Convenience function that handles session management.

    Args:
        pdf_content: Raw PDF bytes
        gcp_path: GCP Cloud Storage path to the PDF
        config: Optional pipeline configuration

    Returns:
        PipelineResult with processing details
    """
    pipeline = MedicalIngestionPipeline(config=config)

    async with get_session_context() as session:
        return await pipeline.process_paper(
            session=session,
            pdf_content=pdf_content,
            gcp_path=gcp_path,
        )
