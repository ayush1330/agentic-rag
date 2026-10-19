"""
Paper Management Service

Provides high-level operations for managing medical research papers,
including deletion with coordinated cleanup across database and GCP storage.
"""

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.infrastructure.database.models import Paper, PaperChunk
from medical_agent.infrastructure.gcp_storage import GCPStorageClient, get_storage_client

logger = logging.getLogger(__name__)


@dataclass
class DeletionResult:
    """Result of paper deletion operation."""

    paper_id: uuid.UUID
    paper_title: str
    deleted_from_db: bool
    deleted_from_gcp: bool
    chunks_deleted: int
    gcp_path: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if deletion was fully successful."""
        return self.deleted_from_db and self.deleted_from_gcp

    @property
    def partial_success(self) -> bool:
        """Check if deletion was partially successful."""
        return self.deleted_from_db or self.deleted_from_gcp


class PaperManager:
    """
    High-level service for managing medical research papers.

    Coordinates operations across database and cloud storage to ensure
    data consistency.
    """

    def __init__(self, storage_client: GCPStorageClient | None = None) -> None:
        """
        Initialize the paper manager.

        Args:
            storage_client: GCP storage client (uses default if not provided)
        """
        self.storage_client = storage_client or get_storage_client()

    async def delete_paper(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        delete_from_gcp: bool = True,
        commit: bool = True,
    ) -> DeletionResult:
        """
        Delete a paper and all associated data.

        Deletion order (reversible operations first, irreversible last):
        1. Retrieves paper metadata from database
        2. Counts associated chunks (for reporting)
        3. Deletes chunks from database (explicit DELETE, paper_id in JSONB)
        4. Deletes paper record from database
        5. Commits DB transaction
        6. Deletes PDF from GCP Cloud Storage (LAST - irreversible)

        If GCP deletion fails after DB commit, paper is still removed from DB.
        Orphan file in GCP is acceptable (wasted storage, not data inconsistency).

        Args:
            session: Active database session
            paper_id: UUID of the paper to delete
            delete_from_gcp: Whether to also delete the PDF from GCP (default: True)
            commit: Whether to commit the transaction (default: True)

        Returns:
            DeletionResult with details about what was deleted

        Example:
            async with get_session_context() as session:
                result = await paper_manager.delete_paper(session, paper_id)
                if result.success:
                    print(f"Deleted '{result.paper_title}' and {result.chunks_deleted} chunks")
                else:
                    print(f"Partial deletion: {result.error}")
        """
        logger.info(f"Starting deletion for paper_id: {paper_id}")

        # Step 1: Retrieve paper metadata (idempotency check)
        paper = await self._get_paper(session, paper_id)
        if paper is None:
            logger.info(f"Paper {paper_id} already deleted or does not exist")
            # Return idempotent success result
            return DeletionResult(
                paper_id=paper_id,
                paper_title="<already deleted>",
                deleted_from_db=True,
                deleted_from_gcp=True,
                chunks_deleted=0,
                gcp_path=None,
                error=None,
            )

        paper_title = paper.title
        gcp_path = paper.gcp_path

        # Step 2: Count chunks before deletion (for reporting)
        chunk_count = await self._count_paper_chunks(session, paper_id)
        logger.info(f"Found {chunk_count} chunks for paper '{paper_title}'")

        deleted_from_db = False
        deleted_from_gcp = False
        error_msg = None

        # Step 3: Delete chunks first (reversible via rollback)
        try:
            from sqlalchemy import delete

            # Delete all chunks for this paper
            delete_chunks_stmt = delete(PaperChunk).where(
                PaperChunk.metadata_["paper_id"].astext == str(paper_id)
            )
            await session.execute(delete_chunks_stmt)
            logger.info(f"Deleted {chunk_count} chunks for paper '{paper_title}'")

        except Exception as e:
            error_msg = f"Chunk deletion failed: {str(e)}"
            logger.error(error_msg)
            await session.rollback()
            raise

        # Step 4: Delete paper record (reversible via rollback)
        try:
            await session.delete(paper)

            if commit:
                await session.commit()
                logger.info(f"Deleted paper '{paper_title}' from database")
            else:
                await session.flush()
                logger.info(f"Marked paper '{paper_title}' for deletion (not committed)")

            deleted_from_db = True

        except Exception as e:
            error_msg = f"Database deletion failed: {str(e)}"
            logger.error(error_msg)
            await session.rollback()
            raise

        # Step 5: Delete from GCP LAST (irreversible, but DB is already clean)
        # At this point DB state is authoritative; GCP cleanup is best-effort.
        if delete_from_gcp and gcp_path:
            try:
                gcp_file_path = self._extract_gcp_path(gcp_path)
                deleted_from_gcp = self.storage_client.delete_pdf(gcp_file_path)

                if deleted_from_gcp:
                    logger.info(f"Deleted PDF from GCP: {gcp_file_path}")
                else:
                    error_msg = f"PDF not found in GCP (may already be deleted): {gcp_file_path}"
                    logger.warning(error_msg)
                    deleted_from_gcp = True  # File doesn't exist = effectively deleted

            except Exception as e:
                # DB is already committed, log error but don't fail
                # Orphan file in GCP is acceptable (wasted storage, not data inconsistency)
                error_msg = f"GCP deletion failed (DB already cleaned): {str(e)}"
                logger.error(error_msg)
                # Don't raise - partial success is acceptable here

        elif not delete_from_gcp:
            logger.info("Skipped GCP deletion (delete_from_gcp=False)")
            # Explicit opt-out is considered successful for this operation mode.
            deleted_from_gcp = True

        elif not gcp_path:
            logger.warning("No GCP path found for paper")
            # Missing storage path means there is nothing actionable to delete.
            deleted_from_gcp = True

        # Return comprehensive result
        # `error` may be populated even when partial_success=True to signal
        # operational follow-up (typically manual GCP cleanup).
        result = DeletionResult(
            paper_id=paper_id,
            paper_title=paper_title,
            deleted_from_db=deleted_from_db,
            deleted_from_gcp=deleted_from_gcp,
            chunks_deleted=chunk_count,
            gcp_path=gcp_path,
            error=error_msg,
        )

        logger.info(
            f"Deletion complete: paper='{paper_title}', "
            f"db={deleted_from_db}, gcp={deleted_from_gcp}, "
            f"chunks={chunk_count}"
        )

        return result

    async def delete_papers_bulk(
        self,
        session: AsyncSession,
        paper_ids: list[uuid.UUID],
        delete_from_gcp: bool = True,
    ) -> list[DeletionResult]:
        """
        Delete multiple papers in a single transaction.

        Args:
            session: Active database session
            paper_ids: List of paper UUIDs to delete
            delete_from_gcp: Whether to delete PDFs from GCP

        Returns:
            List of DeletionResult for each paper (includes idempotent results)

        Note:
            Uses idempotent deletion - papers already deleted are reported as success.
            If any deletion fails (other than paper not found), the entire transaction
            is rolled back. For individual error handling, use delete_paper() separately.
        """
        results = []

        for paper_id in paper_ids:
            try:
                # delete_paper is now idempotent - returns success if already deleted
                result = await self.delete_paper(
                    session,
                    paper_id,
                    delete_from_gcp=delete_from_gcp,
                    commit=False,  # Commit all at once
                )
                results.append(result)
            except Exception as e:
                # Any error - rollback and raise
                logger.error(f"Bulk deletion failed at paper {paper_id}: {e}")
                await session.rollback()
                raise

        # Commit all deletions at once
        await session.commit()
        logger.info(f"Bulk deleted {len(results)} papers")

        # Verify all chunks were deleted
        for result in results:
            if result.deleted_from_db:
                remaining = await self._count_paper_chunks(session, result.paper_id)
                if remaining > 0:
                    logger.error(
                        f"Verification failed: {remaining} chunks remain for paper {result.paper_id}"
                    )
                    result.error = f"Chunks not fully deleted: {remaining} remain"

        return results

    async def get_paper_info(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
    ) -> dict[str, str | int | bool | None] | None:
        """
        Get paper metadata for display/confirmation before deletion.

        Args:
            session: Active database session
            paper_id: UUID of the paper

        Returns:
            Dict with paper info or None if not found
        """
        paper = await self._get_paper(session, paper_id)

        if paper is None:
            return None

        chunk_count = await self._count_paper_chunks(session, paper_id)

        return {
            "id": str(paper.id),
            "title": paper.title,
            "authors": paper.authors,
            "journal": paper.journal,
            "publication_year": paper.publication_year,
            "doi": paper.doi,
            "gcp_path": paper.gcp_path,
            "is_processed": paper.is_processed,
            "chunk_count": chunk_count,
            "created_at": paper.created_at.isoformat() if paper.created_at else None,
        }

    async def list_papers(
        self,
        session: AsyncSession,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        List all papers in the database.

        Args:
            session: Active database session
            limit: Maximum number of papers to return
            offset: Number of papers to skip

        Returns:
            List of paper info dicts
        """
        stmt = select(Paper).order_by(Paper.created_at.desc()).limit(limit).offset(offset)

        result = await session.execute(stmt)
        papers = result.scalars().all()

        paper_list = []
        for paper in papers:
            chunk_count = await self._count_paper_chunks(session, paper.id)
            paper_list.append(
                {
                    "id": str(paper.id),
                    "title": paper.title,
                    "authors": paper.authors,
                    "publication_year": paper.publication_year,
                    "doi": paper.doi,
                    "is_processed": paper.is_processed,
                    "chunk_count": chunk_count,
                    "created_at": paper.created_at.isoformat() if paper.created_at else None,
                }
            )

        return paper_list

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _get_paper(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
    ) -> Paper | None:
        """Retrieve a paper by ID."""
        stmt = select(Paper).where(Paper.id == paper_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _count_paper_chunks(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
    ) -> int:
        """Count chunks associated with a paper (paper_id stored in metadata_ JSONB)."""
        from sqlalchemy import func

        stmt = select(func.count(PaperChunk.id)).where(
            PaperChunk.metadata_["paper_id"].astext == str(paper_id)
        )
        result = await session.execute(stmt)
        return result.scalar() or 0

    def _extract_gcp_path(self, gcp_uri: str) -> str:
        """
        Extract the file path from a GCS URI.

        Args:
            gcp_uri: Full GCS URI (e.g., "gs://bucket_name/path/to/file.pdf")

        Returns:
            Just the path portion (e.g., "path/to/file.pdf")
        """
        if gcp_uri.startswith("gs://"):
            # Remove "gs://" prefix
            path_with_bucket = gcp_uri[5:]
            # Remove bucket name (everything before first "/")
            if "/" in path_with_bucket:
                return path_with_bucket.split("/", 1)[1]
            else:
                return path_with_bucket

        # If it's already just a path, return as-is
        return gcp_uri


# ============================================================================
# Convenience Functions
# ============================================================================


async def delete_paper_by_id(
    session: AsyncSession,
    paper_id: uuid.UUID,
    delete_from_gcp: bool = True,
) -> DeletionResult:
    """
    Convenience function to delete a paper by ID.

    Note: Uses commit=False - caller's context manager handles commit.

    Usage:
        from medical_agent.core.paper_manager import delete_paper_by_id
        from medical_agent.infrastructure.database.session import get_session_context

        async with get_session_context() as session:
            result = await delete_paper_by_id(session, paper_id)
            print(f"Deleted: {result.paper_title}")
    """
    manager = PaperManager()
    return await manager.delete_paper(
        session, paper_id, delete_from_gcp=delete_from_gcp, commit=False
    )


async def delete_paper_by_doi(
    session: AsyncSession,
    doi: str,
    delete_from_gcp: bool = True,
) -> DeletionResult:
    """
    Convenience function to delete a paper by DOI.

    Args:
        session: Active database session
        doi: Digital Object Identifier of the paper
        delete_from_gcp: Whether to delete PDF from GCP

    Returns:
        DeletionResult

    Raises:
        ValueError: If paper with DOI not found
    """
    # Find paper by DOI
    stmt = select(Paper).where(Paper.doi == doi)
    result = await session.execute(stmt)
    paper = result.scalar_one_or_none()

    if paper is None:
        raise ValueError(f"Paper not found with DOI: {doi}")

    # Delete using paper ID (commit=False, let caller's context manager handle it)
    manager = PaperManager()
    return await manager.delete_paper(
        session, paper.id, delete_from_gcp=delete_from_gcp, commit=False
    )
