"""
Paper Deletion Script - Interactive Mode

Delete papers from database and GCP storage with confirmation.

Usage:
    python scripts/paper_deletion.py    # Interactive mode
"""

import asyncio
import logging
import sys
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from medical_agent.core.paper_manager import PaperManager, delete_paper_by_id, delete_paper_by_doi
from medical_agent.infrastructure.database.session import get_session_context


async def list_all_papers() -> list[dict]:
    """List all papers in the database and return the list."""
    print("\n" + "=" * 80)
    print("LISTING ALL PAPERS")
    print("=" * 80)

    manager = PaperManager()
    async with get_session_context() as session:
        papers = await manager.list_papers(session, limit=1000)

        if not papers:
            print("No papers found in the database.")
            return []

        print(f"\nFound {len(papers)} paper(s):\n")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. Title: {paper['title']}")
            print(f"   ID: {paper['id']}")
            print(f"   Authors: {paper['authors']}")
            print(f"   Year: {paper['publication_year']}")
            print(f"   Chunks: {paper['chunk_count']}")
            print()

        return papers


async def get_paper_details(paper_id: str) -> dict | None:
    """Get detailed information about a specific paper."""
    print("\n" + "=" * 80)
    print(f"PAPER DETAILS FOR ID: {paper_id}")
    print("=" * 80)

    manager = PaperManager()
    async with get_session_context() as session:
        paper_info = await manager.get_paper_info(session, uuid.UUID(paper_id))

        if not paper_info:
            print(f"Paper not found: {paper_id}")
            return None

        print(f"\nTitle: {paper_info['title']}")
        print(f"Authors: {paper_info['authors']}")
        print(f"Journal: {paper_info['journal']}")
        print(f"Year: {paper_info['publication_year']}")
        print(f"DOI: {paper_info['doi']}")
        print(f"GCP Path: {paper_info['gcp_path']}")
        print(f"Processed: {paper_info['is_processed']}")
        print(f"Chunks: {paper_info['chunk_count']}")
        print(f"Created: {paper_info['created_at']}")

        return paper_info


async def delete_paper_example(paper_id: str, delete_from_gcp: bool = True) -> None:
    """
    Example: Delete a paper by ID.

    Args:
        paper_id: UUID string of the paper to delete
        delete_from_gcp: Whether to also delete from GCP storage
    """
    print("\n" + "=" * 80)
    print(f"DELETING PAPER: {paper_id}")
    print(f"Delete from GCP: {delete_from_gcp}")
    print("=" * 80)

    try:
        async with get_session_context() as session:
            # Get paper info before deletion (for confirmation)
            manager = PaperManager()
            paper_info = await manager.get_paper_info(session, uuid.UUID(paper_id))

            if not paper_info:
                print(f"\nERROR: Paper not found: {paper_id}")
                return

            print(f"\nPaper to delete:")
            print(f"  Title: {paper_info['title']}")
            print(f"  Chunks: {paper_info['chunk_count']}")
            print(f"  GCP Path: {paper_info['gcp_path']}")

            # Confirm deletion
            confirm = input("\nAre you sure you want to delete this paper? (yes/no): ")
            if confirm.lower() != "yes":
                print("Deletion cancelled.")
                return

            # Delete the paper
            result = await delete_paper_by_id(
                session,
                uuid.UUID(paper_id),
                delete_from_gcp=delete_from_gcp,
            )

            # Display results
            print("\n" + "-" * 80)
            print("DELETION RESULT")
            print("-" * 80)
            print(f"Paper: {result.paper_title}")
            print(f"Deleted from DB: {result.deleted_from_db}")
            print(f"Deleted from GCP: {result.deleted_from_gcp}")
            print(f"Chunks deleted: {result.chunks_deleted}")

            if result.success:
                print("\n✓ SUCCESS: Paper fully deleted!")
            elif result.partial_success:
                print(f"\n⚠ PARTIAL SUCCESS: {result.error}")
            else:
                print(f"\n✗ FAILED: {result.error}")

    except ValueError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


async def delete_all_papers_example() -> None:
    """
    Delete all papers from the database and GCP storage.
    Requires double confirmation for safety.
    """
    print("\n" + "=" * 80)
    print("DELETE ALL PAPERS")
    print("=" * 80)

    manager = PaperManager()

    try:
        async with get_session_context() as session:
            # List all papers first
            papers = await manager.list_papers(session, limit=1000)

            if not papers:
                print("\nNo papers found to delete.")
                return

            print(f"\nAbout to delete ALL {len(papers)} papers:")
            for i, paper in enumerate(papers[:5], 1):
                print(f"  {i}. {paper['title']} ({paper['id']})")
            if len(papers) > 5:
                print(f"  ... and {len(papers) - 5} more")

            # First confirmation
            confirm1 = input("\nAre you sure? (yes/no): ").strip().lower()
            if confirm1 != "yes":
                print("Deletion cancelled.")
                return

            # Second confirmation for safety
            confirm2 = input("Type 'DELETE ALL' to confirm permanent deletion: ").strip()
            if confirm2 != "DELETE ALL":
                print("Deletion cancelled.")
                return

            # Get all paper IDs
            paper_ids = [uuid.UUID(p['id']) for p in papers]

            # Bulk delete
            results = await manager.delete_papers_bulk(session, paper_ids)

            # Display results
            print("\n" + "-" * 80)
            print("BULK DELETION RESULTS")
            print("-" * 80)

            success_count = sum(1 for r in results if r.success)
            partial_count = sum(1 for r in results if r.partial_success and not r.success)
            failed_count = len(results) - success_count - partial_count

            print(f"\nTotal: {len(results)}")
            print(f"✓ Success: {success_count}")
            print(f"⚠ Partial: {partial_count}")
            print(f"✗ Failed: {failed_count}")

            if failed_count > 0:
                print("\nFailed deletions:")
                for result in results:
                    if not result.success and not result.partial_success:
                        print(f"  ✗ {result.paper_title}: {result.error}")

    except Exception as e:
        print(f"\nDELETION FAILED: {e}")
        import traceback
        traceback.print_exc()


def parse_selected_indices(input_str: str, max_index: int) -> list[int]:
    """Parse comma-separated numbers from user input."""
    try:
        indices = [int(x.strip()) - 1 for x in input_str.split(",")]
        # Validate indices
        if any(i < 0 or i >= max_index for i in indices):
            raise ValueError("Index out of range")
        return indices
    except (ValueError, IndexError):
        return []


async def interactive_mode() -> None:
    """Interactive mode for paper deletion."""
    print("\n" + "=" * 80)
    print("PAPER DELETION - INTERACTIVE MODE")
    print("=" * 80)

    while True:
        print("\nOptions:")
        print("1. List all papers")
        print("2. Delete all papers")
        print("3. Delete by selecting IDs")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            await list_all_papers()

        elif choice == "2":
            # Delete all papers
            manager = PaperManager()
            async with get_session_context() as session:
                papers = await manager.list_papers(session, limit=1000)

                if not papers:
                    print("\nNo papers found to delete.")
                    continue

                print(f"\nAbout to delete ALL {len(papers)} papers:")
                for i, paper in enumerate(papers[:5], 1):
                    print(f"  {i}. {paper['title']}")
                if len(papers) > 5:
                    print(f"  ... and {len(papers) - 5} more")

                # First confirmation
                confirm1 = input("\nAre you sure? (yes/no): ").strip().lower()
                if confirm1 != "yes":
                    print("Deletion cancelled.")
                    continue

                # Second confirmation for safety
                confirm2 = input("Type 'DELETE ALL' to confirm permanent deletion: ").strip()
                if confirm2 != "DELETE ALL":
                    print("Deletion cancelled.")
                    continue

                # Get all paper IDs
                paper_ids = [uuid.UUID(p['id']) for p in papers]

                # Bulk delete
                results = await manager.delete_papers_bulk(session, paper_ids)

                # Display results
                print("\n" + "-" * 80)
                print("BULK DELETION RESULTS")
                print("-" * 80)

                success_count = sum(1 for r in results if r.success)
                partial_count = sum(1 for r in results if r.partial_success and not r.success)
                failed_count = len(results) - success_count - partial_count

                print(f"\nTotal: {len(results)}")
                print(f"✓ Success: {success_count}")
                print(f"⚠ Partial: {partial_count}")
                print(f"✗ Failed: {failed_count}")

        elif choice == "3":
            # Delete by selecting IDs
            papers = await list_all_papers()

            if not papers:
                print("No papers to delete.")
                continue

            ids_input = input("\nEnter paper IDs to delete (e.g., 1,3,5,8): ").strip()
            selected_indices = parse_selected_indices(ids_input, len(papers))

            if not selected_indices:
                print("Invalid input. Please enter valid paper numbers.")
                continue

            selected_papers = [papers[i] for i in selected_indices]

            # Show preview
            print("\n" + "-" * 80)
            print(f"Papers to delete ({len(selected_papers)} selected):")
            print("-" * 80)
            for paper in selected_papers:
                print(f"  • {paper['title']} (ID: {paper['id']})")
            print("-" * 80)

            # First confirmation
            confirm1 = input(f"\nDelete {len(selected_papers)} papers? (yes/no): ").strip().lower()
            if confirm1 != "yes":
                print("Deletion cancelled.")
                continue

            # Second confirmation for safety
            confirm2 = input("Type 'DELETE' to confirm deletion: ").strip()
            if confirm2 != "DELETE":
                print("Deletion cancelled.")
                continue

            # Bulk delete selected papers
            manager = PaperManager()
            async with get_session_context() as session:
                paper_ids = [uuid.UUID(p['id']) for p in selected_papers]
                results = await manager.delete_papers_bulk(session, paper_ids)

                # Display results
                print("\n" + "-" * 80)
                print("DELETION RESULTS")
                print("-" * 80)

                success_count = sum(1 for r in results if r.success)
                partial_count = sum(1 for r in results if r.partial_success and not r.success)
                failed_count = len(results) - success_count - partial_count

                print(f"\nTotal: {len(results)}")
                print(f"✓ Success: {success_count}")
                print(f"⚠ Partial: {partial_count}")
                print(f"✗ Failed: {failed_count}")

        elif choice == "4":
            print("\nExiting...")
            break

        else:
            print("\nInvalid choice. Please try again.")


async def main() -> None:
    """Main entry point."""
    try:
        await interactive_mode()
    except KeyboardInterrupt:
        print("\nDeletion cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
