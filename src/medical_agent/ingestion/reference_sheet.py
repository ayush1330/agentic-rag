"""
Reference sheet loader for NLM-style citations (Part 0).

Reads an external Excel workbook mapping paper filename → NLM-style
citation text. Used by the ingestion script to attach a citation to each
paper before it runs through the pipeline.

Part 0 is a throwaway stepping stone: the citation text lives in the
LlamaIndex chunk metadata JSONB column, not in the `papers` SQL schema.
No database migrations.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_PDF_SUFFIX_RE = re.compile(r"\.pdf$", re.IGNORECASE)

MATCHING_RESULTS_SHEET = "Full DB_v2_503_pdfs"
FILENAME_COLUMN = "A"
CITATION_COLUMN = "F"
HEADER_ROW = 1

_FILENAME_INDEX = column_index_from_string(FILENAME_COLUMN) - 1
_CITATION_INDEX = column_index_from_string(CITATION_COLUMN) - 1

# Columns verified against the v2 .xlsx on 2026-06-08.
# Layout: A=filename, B=DOI, C=PMID, D=PMCID, E=NLM citation style (unused),
# F=NLM reference text (without title), G=publication title, H=comments (unused).
DOI_COLUMN = "B"
PMID_COLUMN = "C"
PMCID_COLUMN = "D"
TITLE_COLUMN = "G"

_PMID_INDEX = column_index_from_string(PMID_COLUMN) - 1
_PMCID_INDEX = column_index_from_string(PMCID_COLUMN) - 1
_DOI_INDEX = column_index_from_string(DOI_COLUMN) - 1
_TITLE_INDEX = column_index_from_string(TITLE_COLUMN) - 1


@dataclass
class PaperReference:
    """Structured record parsed from one row of the reference sheet."""

    paper_name: str          # normalized gcp_file_name (lookup key)
    nlm_reference: str       # full NLM citation text (required)
    title: str               # publication title (required, maps to papers.title)
    pmid: str | None = None  # PubMed ID (optional)
    pmcid: str | None = None # PubMed Central ID (optional)
    doi: str | None = None   # DOI (optional)


def normalize_filename(name: str) -> str:
    """
    Normalize a paper filename for dictionary lookup.

    Applies four rules, in order:
    1. Strip leading/trailing whitespace.
    2. Remove a trailing `.pdf` or `.PDF` extension.
    3. Lowercase.
    4. Collapse consecutive whitespace to a single space.

    The same normalizer is applied to both the Excel sheet's column A
    values and to GCP object names during lookup. Any mismatch after
    normalization indicates a real data-entry divergence that the
    pre-flight coverage check will report.
    """
    stripped = name.strip()
    unescaped = html.unescape(stripped)
    without_ext = _PDF_SUFFIX_RE.sub("", unescaped)
    lowered = without_ext.lower()
    collapsed = _WHITESPACE_RE.sub(" ", lowered).strip()
    return collapsed


def load_reference_map(sheet_path: Path) -> dict[str, PaperReference]:
    """
    Load the external Excel reference sheet into a normalized lookup dict.

    Reads the `Full DB_v2_503_pdfs` tab, iterates from row 2 (row 1 is headers),
    and builds a dict mapping `normalize_filename(col A)` → `PaperReference`. Rows
    where filename, citation, or title is empty are logged at WARNING level and skipped.

    Raises:
        FileNotFoundError: if `sheet_path` does not exist.
        ValueError: if the workbook does not contain a `Full DB_v2_503_pdfs` sheet.
    """
    if not sheet_path.exists():
        raise FileNotFoundError(f"Reference sheet not found: {sheet_path}")

    logger.info("Loading reference sheet: %s", sheet_path)
    workbook = load_workbook(filename=sheet_path, read_only=True, data_only=True)

    try:
        if MATCHING_RESULTS_SHEET not in workbook.sheetnames:
            raise ValueError(
                f"Reference sheet missing required tab '{MATCHING_RESULTS_SHEET}'. "
                f"Found tabs: {workbook.sheetnames}"
            )

        sheet = workbook[MATCHING_RESULTS_SHEET]
        ref_map: dict[str, PaperReference] = {}
        total_rows = 0
        skipped = 0

        for idx, row in enumerate(sheet.iter_rows(min_row=HEADER_ROW + 1, values_only=True)):
            actual_row = idx + HEADER_ROW + 1
            total_rows += 1
            filename = row[_FILENAME_INDEX] if len(row) > _FILENAME_INDEX else None
            citation = row[_CITATION_INDEX] if len(row) > _CITATION_INDEX else None
            title = row[_TITLE_INDEX] if len(row) > _TITLE_INDEX else None
            pmid_raw = row[_PMID_INDEX] if len(row) > _PMID_INDEX else None
            pmcid_raw = row[_PMCID_INDEX] if len(row) > _PMCID_INDEX else None
            doi_raw = row[_DOI_INDEX] if len(row) > _DOI_INDEX else None

            if not filename or not citation:
                skipped += 1
                logger.warning(
                    "Skipping sheet row %d: filename=%r citation=%r",
                    actual_row,
                    filename,
                    citation,
                )
                continue

            if not title:
                skipped += 1
                logger.warning(
                    "Skipping sheet row %d: title is empty (papers.title is NOT NULL)",
                    actual_row,
                )
                continue

            key = normalize_filename(str(filename))
            if not key:
                skipped += 1
                logger.warning(
                    "Skipping sheet row %d: filename normalizes to empty string (%r)",
                    actual_row,
                    filename,
                )
                continue

            ref_map[key] = PaperReference(
                paper_name=key,
                nlm_reference=str(citation).strip(),
                title=str(title).strip(),
                pmid=(str(pmid_raw).strip() or None) if pmid_raw is not None else None,
                pmcid=(str(pmcid_raw).strip() or None) if pmcid_raw is not None else None,
                doi=(str(doi_raw).strip() or None) if doi_raw is not None else None,
            )

        logger.info(
            "Reference sheet loaded: %d rows, %d entries, %d skipped",
            total_rows,
            len(ref_map),
            skipped,
        )
        return ref_map
    finally:
        workbook.close()


def check_citation_coverage(
    gcp_paths: Iterable[str],
    ref_map: dict[str, PaperReference],
) -> list[str]:
    """
    Return the GCP paths whose normalized filename stems have no entry in
    `ref_map`. Used as a hard pre-flight check before ingestion begins.

    The function takes the filename stem (the last path component with the
    `.pdf` extension removed) and runs it through `normalize_filename`
    before looking it up. Returns the original (un-normalized) paths so
    the caller can print a useful error list.
    """
    missing: list[str] = []
    for path in gcp_paths:
        stem = Path(path).stem
        key = normalize_filename(stem)
        if key not in ref_map:
            missing.append(path)
    return missing
