"""
Chunk quality filter using Docling structural metadata.

Uses Docling's `headings` and `doc_items` labels for reliable section filtering,
plus lightweight heuristics for noise that Docling labels don't catch:
- Dots-only or whitespace-heavy noise
- Low alpha ratio (equations, symbols)
- Very short chunks (< 15 words)
"""

import logging
import re

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

# Sections to drop (matched against Docling headings)
EXCLUDED_SECTIONS = {
    "reference",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "supplementary",
    "supplemental",
    "conflict of interest",
    "funding",
    "author contributions",
    "abbreviations",
}

# Docling doc_item labels to drop
EXCLUDED_LABELS = {"reference", "page_header", "page_footer", "document_index"}

# Noise patterns for content-level checks (things Docling labels won't catch)
NOISE_PATTERNS = [
    re.compile(r"^[\s\.·•\-_=]{10,}$", re.IGNORECASE | re.MULTILINE),  # Dots, bullets, lines
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE),  # Page numbers
    re.compile(r"^[^\w]*$", re.IGNORECASE | re.MULTILINE),  # No word characters
]

# Short chunks with these patterns are header/footer garbage
HEADER_FOOTER_PATTERNS = [
    re.compile(r"journal\s+of\s+\w+", re.IGNORECASE),
    re.compile(r"copyright\s+©?\s*\d{4}", re.IGNORECASE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"downloaded\s+from", re.IGNORECASE),
]


def _is_excluded_section(node: BaseNode) -> bool:
    """Check if chunk belongs to an excluded section using Docling headings."""
    headings = node.metadata.get("headings") or []
    for heading in headings:
        if not isinstance(heading, str):
            continue
        heading_lower = heading.strip().lower()
        if any(excluded in heading_lower for excluded in EXCLUDED_SECTIONS):
            return True
    return False


def _has_excluded_labels(node: BaseNode) -> bool:
    """Check if chunk's doc_items are all excluded Docling labels."""
    doc_items = node.metadata.get("doc_items") or []
    if not doc_items:
        return False

    labels = set()
    for item in doc_items:
        if isinstance(item, dict):
            label = (item.get("label") or "").lower()
        else:
            label = getattr(item, "label", "")
            if hasattr(label, "value"):
                label = label.value
            label = str(label).lower()
        if label:
            labels.add(label)

    return bool(labels) and labels.issubset(EXCLUDED_LABELS)


def is_low_quality_chunk(node: BaseNode, min_words: int = 15) -> bool:
    """
    Check if a chunk should be filtered out.

    Uses Docling structural metadata first (reliable), then falls back
    to content heuristics for noise detection.

    Returns:
        True if chunk should be REMOVED
    """
    text = node.get_content().strip()
    word_count = len(text.split())

    # Too short to be useful
    if word_count < min_words:
        return True

    # Docling structural filters (reliable)
    if _is_excluded_section(node):
        return True

    if _has_excluded_labels(node):
        return True

    # Content heuristics (catches what Docling labels miss)

    # Majority noise lines
    lines = text.split("\n")
    noise_lines = sum(1 for line in lines if any(p.match(line.strip()) for p in NOISE_PATTERNS))
    if noise_lines / max(len(lines), 1) > 0.5:
        return True

    # Header/footer garbage (short chunks only)
    if word_count < 30 and any(p.search(text) for p in HEADER_FOOTER_PATTERNS):
        return True

    # Equation/symbol heavy (low readable text ratio)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return True

    return False


def filter_chunks(nodes: list[BaseNode]) -> list[BaseNode]:
    """
    Filter out low-quality chunks.

    Args:
        nodes: List of chunk nodes from DoclingNodeParser

    Returns:
        Filtered list with structural garbage and noise removed
    """
    original_count = len(nodes)
    filtered = [n for n in nodes if not is_low_quality_chunk(n)]
    removed = original_count - len(filtered)

    if removed > 0:
        logger.info(
            f"Chunk filter: {original_count} → {len(filtered)} (removed {removed}, {removed / original_count * 100:.1f}%)"
        )

    return filtered
