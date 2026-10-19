#!/usr/bin/env python3
"""
PubMed Bulk Paper Downloader (v3)

Downloads open-access PDFs for papers listed in an Excel file.
Strategy: Resolve DOIs + PMCIDs upfront, then 5-source fallback chain:
  direct_pmc → pmc_oa → unpaywall → openalex → europe_pmc

Fixes over v2:
  - DOI normalization (strips URL prefixes)
  - Direct PMC PDF download (highest reliability source)
  - All PDF URLs tried per source (not just the first)
  - Structured logging to file + console
  - Timestamped CSV reports
  - Canonical filenames (PMID_{pmid}.pdf) with legacy compat
  - No stream=True (avoids truncated downloads)

Usage:
    python scripts/pubmed_download_paper.py
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import openpyxl
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_EXCEL = "/Users/ayushsingh/Library/CloudStorage/OneDrive-Personal/RAG/pubmed_bulk_download.xlsx"
OUTPUT_DIR = "/Users/ayushsingh/Library/CloudStorage/OneDrive-Personal/RAG/pubmed_bulk_papers"
SHEET_NAMES = [
    "Vaginal pH and pregnancy list",
    "Ethnicity",
    "Menopause",
]
EMAIL = "ayushsingh@phera.digital"
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")

# ---------------------------------------------------------------------------
# API URLs
# ---------------------------------------------------------------------------

NCBI_ID_CONVERTER_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
OPENALEX_API_URL = "https://api.openalex.org/works"
UNPAYWALL_API_URL = "https://api.unpaywall.org/v2"
CROSSREF_API_URL = "https://api.crossref.org/works"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = f"phera_paper_downloader/3.0 (mailto:{EMAIL})"
REQUEST_TIMEOUT = 30
MIN_PDF_SIZE = 5 * 1024  # 5 KB
PDF_MAGIC = b"%PDF-"
MAX_RETRIES = 3
BATCH_SIZE = 200

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

REPORT_COLUMNS = [
    "sheet",
    "pmid",
    "doi",
    "pmcid",
    "title",
    "status",
    "source",
    "filename",
    "error",
    "attempted_sources",
    "download_time",
]

# Module-level logger and session — initialized in main()
logger = logging.getLogger("pubmed_downloader")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> None:
    """Configure logging to both console and a log file in the output directory."""
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler — INFO level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — DEBUG level (captures everything)
    log_path = output_dir / "download.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Logging initialized — log file: %s", log_path)


# ---------------------------------------------------------------------------
# DOI normalization
# ---------------------------------------------------------------------------

_DOI_PREFIX_RE = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:|DOI:)\s*", re.IGNORECASE
)


def normalize_doi(raw: str | None) -> str | None:
    """
    Strip URL prefixes from a DOI string, returning the bare DOI (e.g. '10.1234/x').

    Returns None if the input is empty or not a valid DOI after stripping.
    """
    if not raw:
        return None
    cleaned = _DOI_PREFIX_RE.sub("", raw.strip())
    if not cleaned or not cleaned.startswith("10."):
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_input(path: str, sheet_names: list[str]) -> list[dict]:
    """Load papers from multiple Excel sheets.

    Returns a list of paper dicts, each tagged with its sheet name.
    Column detection is case-insensitive. DOIs are normalized during loading.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    available = wb.sheetnames

    all_papers: list[dict] = []
    for sheet_name in sheet_names:
        if sheet_name not in available:
            logger.warning(
                "Sheet '%s' not found. Available: %s",
                sheet_name,
                ", ".join(available),
            )
            continue
        papers = _load_sheet(wb, sheet_name)
        logger.info("Sheet '%s': %d papers loaded", sheet_name, len(papers))
        all_papers.extend(papers)

    wb.close()
    return all_papers


def _load_sheet(wb: openpyxl.Workbook, sheet_name: str) -> list[dict]:
    """Load papers from a single Excel sheet.

    Detects columns by header names (case-insensitive):
      pmid, doi, title/publication name, first_author/1st surname, journal, year.
    """
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(min_row=1, values_only=True))

    if not rows:
        logger.warning("Sheet '%s' is empty, skipping.", sheet_name)
        return []

    # Detect column indices from header row
    header = [str(c).strip().lower() if c else "" for c in rows[0]]
    pmid_col: int | None = None
    doi_col: int | None = None
    title_col: int | None = None
    author_col: int | None = None
    journal_col: int | None = None
    year_col: int | None = None

    for i, h in enumerate(header):
        if h == "pmid":
            pmid_col = i
        elif h == "doi":
            doi_col = i
        elif h in ("title", "publication name"):
            title_col = i
        elif h in ("first_author", "1st surname"):
            author_col = i
        elif h == "journal":
            journal_col = i
        elif h == "year":
            year_col = i

    if pmid_col is None:
        logger.warning("No 'pmid' column in sheet '%s', skipping.", sheet_name)
        return []

    def _cell(row: tuple, col: int | None) -> str | None:
        if col is None or col >= len(row):
            return None
        val = row[col]
        if val is None:
            return None
        s = str(val).strip()
        return s if s and s.upper() not in ("N/A", "NA", "NONE") else None

    papers: list[dict] = []
    for row in rows[1:]:
        pmid_val = row[pmid_col] if pmid_col < len(row) else None
        if not pmid_val:
            continue
        pmid = str(int(pmid_val)) if isinstance(pmid_val, float) else str(pmid_val).strip()
        if not pmid or not pmid.isdigit():
            continue

        raw_doi = _cell(row, doi_col)
        doi = normalize_doi(raw_doi)

        title = _cell(row, title_col)
        first_author = _cell(row, author_col)
        journal = _cell(row, journal_col)

        year = None
        if year_col is not None and year_col < len(row):
            raw = row[year_col]
            if raw:
                year = str(int(raw)) if isinstance(raw, float) else str(raw).strip()

        papers.append({
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "first_author": first_author,
            "journal": journal,
            "year": year,
            "sheet": sheet_name,
        })

    return papers


# ---------------------------------------------------------------------------
# DOI resolution for papers missing DOIs
# ---------------------------------------------------------------------------


def resolve_missing_dois(papers: list[dict]) -> int:
    """Resolve DOIs for papers that don't have one.

    Two-step approach:
      1. PubMed EFetch (batch) — fast, authoritative
      2. CrossRef title+author search (fallback) — slower, needs validation

    Mutates papers in-place. Returns count of newly resolved DOIs.
    """
    missing = [p for p in papers if not p["doi"]]
    if not missing:
        logger.info("All papers already have DOIs — skipping resolution.")
        return 0

    resolved_count = 0

    # Step 1: PubMed EFetch batch
    pmids = [p["pmid"] for p in missing]
    logger.info("EFetch: resolving DOIs for %d papers...", len(pmids))

    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i : i + BATCH_SIZE]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }

        resp = _request_with_retry("GET", NCBI_EFETCH_URL, params=params)
        if resp is None:
            logger.warning(
                "EFetch batch failed for PMIDs %s–%s", batch[0], batch[-1]
            )
            continue

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            logger.warning("EFetch returned invalid XML")
            continue

        # Build pmid -> doi map from the response
        missing_by_pmid = {p["pmid"]: p for p in missing if not p["doi"]}
        for article in root.iter("PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                continue
            article_pmid = pmid_elem.text.strip()

            doi_elem = article.find('.//ArticleId[@IdType="doi"]')
            if doi_elem is None or not doi_elem.text:
                continue
            doi = normalize_doi(doi_elem.text.strip())
            if not doi:
                continue

            p = missing_by_pmid.get(article_pmid)
            if p and not p["doi"]:
                p["doi"] = doi
                resolved_count += 1

        if i + BATCH_SIZE < len(pmids):
            time.sleep(0.4)

    efetch_count = resolved_count
    logger.info("EFetch: recovered %d DOIs", efetch_count)

    # Step 2: CrossRef search for any still missing
    still_missing = [p for p in papers if not p["doi"] and p.get("title")]
    if still_missing:
        logger.info(
            "CrossRef: searching for %d remaining papers...", len(still_missing)
        )
        crossref_count = 0
        for p in still_missing:
            doi = _crossref_search_doi(p["title"], p.get("first_author"))
            if doi:
                p["doi"] = doi
                crossref_count += 1
                resolved_count += 1
            time.sleep(0.5)
        logger.info("CrossRef: recovered %d DOIs", crossref_count)

    return resolved_count


def _crossref_search_doi(title: str, author: str | None) -> str | None:
    """Search CrossRef by title (+author) to find a DOI.

    Validates match quality by:
      - Score ratio: top result score / second result score >= 1.3
      - Author surname match: first author's surname must appear in the result

    Returns normalized DOI or None.
    """
    params: dict[str, str] = {
        "query.bibliographic": title,
        "rows": "3",
        "mailto": EMAIL,
    }
    if author:
        params["query.author"] = author

    resp = _request_with_retry("GET", CROSSREF_API_URL, params=params)
    if resp is None:
        return None

    try:
        data = resp.json()
    except ValueError:
        return None

    items = data.get("message", {}).get("items", [])
    if not items:
        return None

    top = items[0]
    top_score = top.get("score", 0)

    # Validate: top result should have a meaningfully higher score than #2
    if len(items) > 1:
        second_score = items[1].get("score", 0)
        if second_score > 0 and top_score / second_score < 1.3:
            logger.debug(
                "CrossRef: ambiguous match for '%s' (score ratio %.2f)",
                title[:60],
                top_score / second_score,
            )
            return None

    # Validate: author surname should appear in the result
    if author:
        result_authors = top.get("author", [])
        author_lower = author.lower()
        found = any(
            author_lower in a.get("family", "").lower() for a in result_authors
        )
        if not found:
            logger.debug(
                "CrossRef: author mismatch for '%s' (expected '%s')",
                title[:60],
                author,
            )
            return None

    return normalize_doi(top.get("DOI"))


# ---------------------------------------------------------------------------
# PMCID resolution
# ---------------------------------------------------------------------------


def resolve_pmcids(papers: list[dict]) -> dict[str, dict]:
    """Batch-resolve PMIDs to PMCIDs via NCBI ID Converter.

    Also picks up DOIs from the response as a bonus supplement.
    Returns dict: pmid -> {"pmcid": str|None, "doi": str|None}
    """
    pmids = [p["pmid"] for p in papers]
    result: dict[str, dict] = {
        p["pmid"]: {"pmcid": None, "doi": p["doi"]} for p in papers
    }

    logger.info("ID Converter: resolving PMCIDs for %d papers...", len(pmids))

    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i : i + BATCH_SIZE]
        params = {
            "ids": ",".join(batch),
            "format": "json",
            "tool": "phera_paper_downloader",
            "email": EMAIL,
        }

        resp = _request_with_retry("GET", NCBI_ID_CONVERTER_URL, params=params)
        if resp is None:
            logger.warning(
                "ID Converter batch failed for PMIDs %s–%s", batch[0], batch[-1]
            )
            continue

        try:
            data = resp.json()
        except ValueError:
            logger.warning("ID Converter returned non-JSON response")
            continue

        for record in data.get("records", []):
            pmid = str(record.get("pmid", ""))
            if not pmid or pmid not in result:
                continue
            pmcid = record.get("pmcid")
            doi = normalize_doi(record.get("doi"))
            if pmcid:
                result[pmid]["pmcid"] = pmcid
            if doi and not result[pmid]["doi"]:
                result[pmid]["doi"] = doi
                # Also update the paper dict
                for p in papers:
                    if p["pmid"] == pmid and not p["doi"]:
                        p["doi"] = doi
                        break

        if i + BATCH_SIZE < len(pmids):
            time.sleep(0.4)

    has_pmcid = sum(1 for v in result.values() if v["pmcid"])
    logger.info("ID Converter: %d/%d papers have PMCIDs", has_pmcid, len(pmids))
    return result


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _request_with_retry(
    method: str,
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    allow_redirects: bool = True,
) -> requests.Response | None:
    """Make an HTTP request with retry on 429/503 and connection errors.

    Uses exponential backoff (1s, 2s, 4s). Max 3 retries.
    No stream=True — downloads full response body to avoid truncation.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.request(
                method,
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers=headers,
                allow_redirects=allow_redirects,
            )
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 503):
                wait = 2**attempt
                logger.debug(
                    "HTTP %d from %s — retrying in %ds (attempt %d/%d)",
                    resp.status_code,
                    url[:80],
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            # 403, 404, etc. — don't retry
            logger.debug(
                "HTTP %d from %s — not retrying", resp.status_code, url[:80]
            )
            return None
        except (requests.ConnectionError, requests.Timeout) as exc:
            wait = 2**attempt
            logger.debug(
                "Connection error for %s: %s — retrying in %ds",
                url[:80],
                exc,
                wait,
            )
            time.sleep(wait)
            continue
    logger.debug("All %d retries exhausted for %s", MAX_RETRIES, url[:80])
    return None


# ---------------------------------------------------------------------------
# PDF validation & saving
# ---------------------------------------------------------------------------


def _save_and_validate_pdf(
    content: bytes, output_path: Path, source: str
) -> bool:
    """Validate that content is a real PDF and save it.

    Checks:
      - PDF magic bytes (%PDF-)
      - Minimum size (5 KB)

    Logs diagnostic info when validation fails.
    """
    if not content:
        logger.debug("[%s] Empty response body", source)
        return False

    # Check for HTML instead of PDF
    if content[:15].lstrip().lower().startswith((b"<!doctype", b"<html")):
        logger.debug(
            "[%s] Got HTML instead of PDF (%d bytes, starts: %s)",
            source,
            len(content),
            content[:80].decode("utf-8", errors="replace"),
        )
        return False

    if not content[:5] == PDF_MAGIC:
        logger.debug(
            "[%s] Bad magic bytes: %s (size %d)",
            source,
            content[:20].decode("utf-8", errors="replace"),
            len(content),
        )
        return False

    if len(content) < MIN_PDF_SIZE:
        logger.debug(
            "[%s] PDF too small: %d bytes (minimum %d)",
            source,
            len(content),
            MIN_PDF_SIZE,
        )
        return False

    output_path.write_bytes(content)
    logger.debug("[%s] Saved %d bytes to %s", source, len(content), output_path.name)
    return True


def _get_output_path(
    pmid: str, pmcid: str | None, output_dir: Path
) -> Path:
    """Determine the canonical output path for a paper.

    Canonical filename: PMID_{pmid}.pdf
    Also checks for legacy v2 format: {pmcid}_{pmid}.pdf (backward compat).
    """
    canonical = output_dir / f"PMID_{pmid}.pdf"
    if canonical.exists():
        return canonical

    # Check for legacy v2 filename
    if pmcid:
        legacy = output_dir / f"{pmcid}_{pmid}.pdf"
        if legacy.exists():
            return legacy

    return canonical


# ---------------------------------------------------------------------------
# Download sources (each returns bool)
# ---------------------------------------------------------------------------


def try_direct_pmc_pdf(pmcid: str, output_path: Path) -> bool:
    """Source 1: Download PDF directly from PMC.

    Constructs URL: https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/
    Uses browser headers to avoid blocks.
    """
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
    logger.debug("[direct_pmc] Trying %s", url)

    resp = _request_with_retry("GET", url, headers=BROWSER_HEADERS)
    if resp is None:
        logger.debug("[direct_pmc] No response for %s", pmcid)
        return False

    return _save_and_validate_pdf(resp.content, output_path, "direct_pmc")


def try_pmc_oa_service(pmcid: str, output_path: Path) -> bool:
    """Source 2: Query PMC OA API for a PDF URL.

    Parses the XML response, extracts the PDF link, converts ftp->https,
    and downloads the PDF. Warns on retracted papers.
    """
    params = {"id": pmcid, "format": "pdf"}
    logger.debug("[pmc_oa] Querying OA service for %s", pmcid)

    resp = _request_with_retry("GET", PMC_OA_URL, params=params)
    if resp is None:
        logger.debug("[pmc_oa] No response from OA service for %s", pmcid)
        return False

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        logger.debug("[pmc_oa] Invalid XML from OA service for %s", pmcid)
        return False

    # Check for error
    error = root.find("error")
    if error is not None:
        logger.debug(
            "[pmc_oa] OA service error for %s: %s",
            pmcid,
            error.text or "unknown",
        )
        return False

    # Check for retraction
    for record in root.iter("record"):
        if record.get("retracted", "no").lower() == "yes":
            logger.warning("[pmc_oa] %s is RETRACTED — skipping", pmcid)
            return False

    # Extract PDF link
    for link in root.iter("link"):
        if link.get("format") == "pdf":
            href = link.get("href", "")
            if not href:
                continue
            # Convert ftp:// to https://
            if href.startswith("ftp://ftp.ncbi.nlm.nih.gov"):
                href = href.replace(
                    "ftp://ftp.ncbi.nlm.nih.gov",
                    "https://ftp.ncbi.nlm.nih.gov",
                )
            logger.debug("[pmc_oa] Downloading from %s", href[:100])
            pdf_resp = _request_with_retry("GET", href, headers=BROWSER_HEADERS)
            if pdf_resp is not None:
                return _save_and_validate_pdf(
                    pdf_resp.content, output_path, "pmc_oa"
                )

    logger.debug("[pmc_oa] No PDF link found for %s", pmcid)
    return False


def try_unpaywall(doi: str, output_path: Path) -> bool:
    """Source 3: Query Unpaywall API for open-access PDF URLs.

    Tries best_oa_location first, then iterates all oa_locations.
    Attempts every url_for_pdf found until one succeeds.
    """
    url = f"{UNPAYWALL_API_URL}/{doi}"
    params = {"email": EMAIL}
    logger.debug("[unpaywall] Querying for DOI %s", doi)

    resp = _request_with_retry("GET", url, params=params)
    if resp is None:
        logger.debug("[unpaywall] No response for DOI %s", doi)
        return False

    try:
        data = resp.json()
    except ValueError:
        logger.debug("[unpaywall] Non-JSON response for DOI %s", doi)
        return False

    # Collect all candidate PDF URLs
    pdf_urls: list[str] = []

    best_loc = data.get("best_oa_location") or {}
    best_pdf = best_loc.get("url_for_pdf")
    if best_pdf:
        pdf_urls.append(best_pdf)

    for loc in data.get("oa_locations", []):
        loc_pdf = loc.get("url_for_pdf")
        if loc_pdf and loc_pdf not in pdf_urls:
            pdf_urls.append(loc_pdf)

    if not pdf_urls:
        logger.debug("[unpaywall] No PDF URLs found for DOI %s", doi)
        return False

    # Try each URL
    for pdf_url in pdf_urls:
        logger.debug("[unpaywall] Trying %s", pdf_url[:100])
        pdf_resp = _request_with_retry("GET", pdf_url, headers=BROWSER_HEADERS)
        if pdf_resp is not None:
            if _save_and_validate_pdf(pdf_resp.content, output_path, "unpaywall"):
                return True

    logger.debug("[unpaywall] All %d URLs failed for DOI %s", len(pdf_urls), doi)
    return False


def try_openalex(doi: str, output_path: Path) -> bool:
    """Source 4: Query OpenAlex API for open-access PDF URLs.

    Uses optional API key from environment. Tries all locations' pdf_url.
    """
    query_url = f"{OPENALEX_API_URL}/doi:{doi}"
    params: dict[str, str] = {"mailto": EMAIL}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    logger.debug("[openalex] Querying for DOI %s", doi)

    resp = _request_with_retry("GET", query_url, params=params)
    if resp is None:
        logger.debug("[openalex] No response for DOI %s", doi)
        return False

    try:
        data = resp.json()
    except ValueError:
        logger.debug("[openalex] Non-JSON response for DOI %s", doi)
        return False

    # Collect all candidate PDF URLs
    pdf_urls: list[str] = []

    best_loc = data.get("best_oa_location") or {}
    best_pdf = best_loc.get("pdf_url")
    if best_pdf:
        pdf_urls.append(best_pdf)

    for loc in data.get("locations", []):
        loc_pdf = loc.get("pdf_url")
        if loc_pdf and loc_pdf not in pdf_urls:
            pdf_urls.append(loc_pdf)

    if not pdf_urls:
        logger.debug("[openalex] No PDF URLs found for DOI %s", doi)
        return False

    # Try each URL
    for pdf_url in pdf_urls:
        logger.debug("[openalex] Trying %s", pdf_url[:100])
        pdf_resp = _request_with_retry("GET", pdf_url, headers=BROWSER_HEADERS)
        if pdf_resp is not None:
            if _save_and_validate_pdf(pdf_resp.content, output_path, "openalex"):
                return True

    logger.debug("[openalex] All %d URLs failed for DOI %s", len(pdf_urls), doi)
    return False


def try_europe_pmc(pmcid: str, output_path: Path) -> bool:
    """Source 5: Download PDF from Europe PMC.

    Tries the ptpmcrender.fcgi endpoint first, then falls back to the
    /articles/{pmcid}?pdf=render endpoint.
    """
    # Attempt 1: ptpmcrender.fcgi
    render_url = "https://europepmc.org/backend/ptpmcrender.fcgi"
    params = {"accid": pmcid, "blobtype": "pdf"}
    logger.debug("[europe_pmc] Trying ptpmcrender for %s", pmcid)

    resp = _request_with_retry("GET", render_url, params=params, headers=BROWSER_HEADERS)
    if resp is not None:
        if _save_and_validate_pdf(resp.content, output_path, "europe_pmc"):
            return True

    # Attempt 2: /articles/{pmcid}?pdf=render
    fallback_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
    logger.debug("[europe_pmc] Trying fallback URL for %s", pmcid)

    resp = _request_with_retry("GET", fallback_url, headers=BROWSER_HEADERS)
    if resp is not None:
        if _save_and_validate_pdf(resp.content, output_path, "europe_pmc"):
            return True

    logger.debug("[europe_pmc] Both attempts failed for %s", pmcid)
    return False


# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------


def process_paper(
    paper: dict, resolved: dict[str, dict], output_dir: Path
) -> dict:
    """Orchestrate the 5-source fallback chain for a single paper.

    Order: direct_pmc -> pmc_oa -> unpaywall -> openalex -> europe_pmc

    Tracks attempted sources for error reporting. Skips already-downloaded
    files for resumability. Returns a result dict for the CSV report.
    """
    pmid = paper["pmid"]
    info = resolved.get(pmid, {"pmcid": None, "doi": paper["doi"]})
    pmcid = info["pmcid"]
    doi = paper["doi"] or info.get("doi")

    output_path = _get_output_path(pmid, pmcid, output_dir)
    filename = output_path.name

    result = {
        "sheet": paper.get("sheet", ""),
        "pmid": pmid,
        "doi": doi or "",
        "pmcid": pmcid or "",
        "title": paper.get("title", ""),
        "status": "failed",
        "source": "",
        "filename": "",
        "error": "",
        "attempted_sources": "",
        "download_time": "",
    }

    # Check if already downloaded (resumability)
    if output_path.exists() and output_path.stat().st_size > MIN_PDF_SIZE:
        result["status"] = "skipped"
        result["source"] = "existing"
        result["filename"] = filename
        result["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.debug("PMID %s: already downloaded (%s)", pmid, filename)
        return result

    attempted: list[str] = []

    # --- Source 1: Direct PMC PDF (requires PMCID) ---
    if pmcid:
        if try_direct_pmc_pdf(pmcid, output_path):
            result["status"] = "downloaded"
            result["source"] = "direct_pmc"
            result["filename"] = filename
            result["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("PMID %s: downloaded via direct_pmc", pmid)
            return result
        attempted.append("direct_pmc:failed")
        time.sleep(0.3)

    # --- Source 2: PMC OA Service (requires PMCID) ---
    if pmcid:
        if try_pmc_oa_service(pmcid, output_path):
            result["status"] = "downloaded"
            result["source"] = "pmc_oa"
            result["filename"] = filename
            result["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("PMID %s: downloaded via pmc_oa", pmid)
            return result
        attempted.append("pmc_oa:failed")
        time.sleep(0.3)

    # --- Source 3: Unpaywall (requires DOI) ---
    if doi:
        if try_unpaywall(doi, output_path):
            result["status"] = "downloaded"
            result["source"] = "unpaywall"
            result["filename"] = filename
            result["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("PMID %s: downloaded via unpaywall", pmid)
            return result
        attempted.append("unpaywall:failed")
        time.sleep(0.3)

    # --- Source 4: OpenAlex (requires DOI) ---
    if doi:
        if try_openalex(doi, output_path):
            result["status"] = "downloaded"
            result["source"] = "openalex"
            result["filename"] = filename
            result["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("PMID %s: downloaded via openalex", pmid)
            return result
        attempted.append("openalex:failed")
        time.sleep(0.3)

    # --- Source 5: Europe PMC (requires PMCID) ---
    if pmcid:
        if try_europe_pmc(pmcid, output_path):
            result["status"] = "downloaded"
            result["source"] = "europe_pmc"
            result["filename"] = filename
            result["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("PMID %s: downloaded via europe_pmc", pmid)
            return result
        attempted.append("europe_pmc:failed")

    # All sources exhausted
    result["attempted_sources"] = ",".join(attempted)
    if not doi and not pmcid:
        result["error"] = "No DOI or PMCID available"
    elif not doi:
        result["error"] = "No DOI (PMC sources failed)"
    elif not pmcid:
        result["error"] = "No PMCID (DOI sources failed)"
    else:
        result["error"] = "All sources exhausted"
    result["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(
        "PMID %s: FAILED — %s (tried: %s)",
        pmid,
        result["error"],
        result["attempted_sources"] or "none",
    )
    return result


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_csv_report(results: list[dict], csv_path: Path) -> None:
    """Write the download report CSV with all result columns."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Report saved: %s", csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full download pipeline.

    Steps:
      1. Validate input file
      2. Create output directory, setup logging
      3. Load papers from all sheets
      4. Resolve missing DOIs (EFetch + CrossRef)
      5. Resolve PMCIDs (NCBI ID Converter)
      6. Download papers with 5-source fallback chain
      7. Write timestamped CSV report
      8. Print summary with source and per-sheet breakdowns
    """
    input_path = Path(INPUT_EXCEL)
    output_dir = Path(OUTPUT_DIR)

    # Validate input
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")
    if input_path.suffix != ".xlsx":
        sys.exit("Input file must be an .xlsx Excel file.")

    # Check if Excel file is readable
    try:
        wb = openpyxl.load_workbook(str(input_path))
        wb.close()
    except PermissionError:
        sys.exit(
            "Cannot access the Excel file. Possible causes:\n"
            "  1. The file is open in Excel/Numbers — close it and retry.\n"
            "  2. macOS is blocking access to OneDrive — go to:\n"
            "     System Settings > Privacy & Security > Full Disk Access\n"
            "     and enable it for your Terminal app, then restart Terminal."
        )

    # Create output directory and setup logging
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("PubMed Bulk Paper Downloader v3")
    logger.info("=" * 60)

    # Load papers from all sheets
    logger.info("Loading papers from %d sheets...", len(SHEET_NAMES))
    papers = load_input(str(input_path), SHEET_NAMES)
    logger.info("Total: %d papers with valid PMIDs", len(papers))

    if not papers:
        sys.exit("No valid papers found in the Excel file.")

    # Resolve missing DOIs
    doi_count_before = sum(1 for p in papers if p["doi"])
    no_doi_count = sum(1 for p in papers if not p["doi"])
    logger.info(
        "DOI status: %d/%d have DOIs, %d missing",
        doi_count_before,
        len(papers),
        no_doi_count,
    )

    if no_doi_count > 0:
        logger.info("Resolving missing DOIs...")
        newly_resolved = resolve_missing_dois(papers)
        doi_count_after = sum(1 for p in papers if p["doi"])
        still_no_doi = sum(1 for p in papers if not p["doi"])
        logger.info(
            "DOI resolution: %d -> %d (resolved %d, still missing %d)",
            doi_count_before,
            doi_count_after,
            newly_resolved,
            still_no_doi,
        )

    # Resolve PMCIDs
    logger.info("Resolving PMCIDs...")
    resolved = resolve_pmcids(papers)

    has_pmcid = sum(1 for v in resolved.values() if v["pmcid"])
    has_doi = sum(1 for p in papers if p["doi"])
    no_path = sum(
        1
        for p in papers
        if not p["doi"]
        and not resolved.get(p["pmid"], {}).get("pmcid")
        and not resolved.get(p["pmid"], {}).get("doi")
    )
    logger.info(
        "Coverage: %d with PMCID, %d with DOI, %d with neither",
        has_pmcid,
        has_doi,
        no_path,
    )

    # Download papers
    logger.info("Downloading PDFs to: %s", output_dir)
    results: list[dict] = []
    downloaded = 0
    skipped = 0
    failed = 0

    for paper in tqdm(papers, desc="Downloading", unit="paper"):
        result = process_paper(paper, resolved, output_dir)
        results.append(result)

        if result["status"] == "downloaded":
            downloaded += 1
            time.sleep(0.3)  # Rate limit between successful downloads
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Downloaded: %d", downloaded)
    logger.info("Skipped:    %d (already existed)", skipped)
    logger.info("Failed:     %d", failed)
    logger.info("=" * 60)

    # Write timestamped CSV report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_path = output_dir / f"download_report_{timestamp}.csv"
    write_csv_report(results, csv_path)

    # Source breakdown
    sources: dict[str, int] = {}
    for r in results:
        if r["status"] == "downloaded":
            src = r["source"]
            sources[src] = sources.get(src, 0) + 1
    if sources:
        logger.info("Download sources:")
        for src, count in sorted(sources.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d", src, count)

    # Per-sheet breakdown
    sheet_stats: dict[str, dict[str, int]] = {}
    for r in results:
        s = r.get("sheet", "unknown")
        if s not in sheet_stats:
            sheet_stats[s] = {"downloaded": 0, "skipped": 0, "failed": 0}
        sheet_stats[s][r["status"]] = sheet_stats[s].get(r["status"], 0) + 1
    if len(sheet_stats) > 1:
        logger.info("Per-sheet breakdown:")
        for sheet, stats in sheet_stats.items():
            logger.info(
                "  %s: %d downloaded, %d skipped, %d failed",
                sheet,
                stats.get("downloaded", 0),
                stats.get("skipped", 0),
                stats.get("failed", 0),
            )

    # Print failed papers (first 20)
    failed_results = [r for r in results if r["status"] == "failed"]
    if failed_results:
        show_count = min(len(failed_results), 20)
        logger.info("Failed papers (%d total, showing first %d):", len(failed_results), show_count)
        for r in failed_results[:show_count]:
            logger.info(
                "  PMID %s: %s (tried: %s)",
                r["pmid"],
                r["error"],
                r["attempted_sources"] or "none",
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
