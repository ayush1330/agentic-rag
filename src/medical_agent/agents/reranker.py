"""Cross-encoder reranker for retrieved nodes."""

import logging
from typing import Any

from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Module-level singleton (loaded once, ~200MB model)
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Get or create the cross-encoder reranker (singleton)."""
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder model: ms-marco-MiniLM-L-6-v2")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info("Cross-encoder loaded")
    return _reranker


def rerank_nodes(
    query: str,
    nodes: list[NodeWithScore],
    top_k: int = 5,
) -> list[NodeWithScore]:
    """
    Rerank retrieved nodes using a cross-encoder.

    Args:
        query: Original user query (not enhanced)
        nodes: Retrieved nodes from hybrid search
        top_k: Number of top results to keep after reranking

    Returns:
        Top-k nodes reranked by cross-encoder score
    """
    if not nodes:
        return nodes

    reranker = get_reranker()

    # Create query-document pairs for cross-encoder
    pairs = [(query, node.get_content()) for node in nodes]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach reranker scores and sort
    for node, score in zip(nodes, scores, strict=False):
        node.score = float(score)  # Replace hybrid score with reranker score

    # Sort by reranker score (descending) and keep top_k
    reranked = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[:top_k]

    logger.info(
        f"Reranked {len(nodes)} → {len(reranked)} nodes. "
        f"Top score: {reranked[0].score:.3f}, Bottom: {reranked[-1].score:.3f}"
    )

    return reranked


def compute_metadata_overlap_score(
    user_profile: dict[str, Any],
    paper_metadata: dict[str, Any],
) -> float:
    """
    Compute overlap between user health profile and paper metadata.

    Uses Jaccard similarity (intersection / union) for each metadata field,
    then averages across all fields to get a 0.0-1.0 overlap score.
    """
    if not user_profile or not paper_metadata:
        return 0.0

    fields_to_check = [
        "diagnoses",
        "symptoms",
        "birth_control",
        "hormone_therapy",
        "ethnicities",
        "menstrual_status",
        "fertility_treatments",
    ]

    field_scores = []

    for field in fields_to_check:
        user_values = set(user_profile.get(field, []))
        paper_values = set(paper_metadata.get(field, []))

        if not user_values and not paper_values:
            continue

        if not user_values or not paper_values:
            field_scores.append(0.0)
            continue

        intersection = len(user_values & paper_values)
        union = len(user_values | paper_values)
        field_scores.append(intersection / union if union > 0 else 0.0)

    return sum(field_scores) / len(field_scores) if field_scores else 0.0


def rerank_nodes_with_metadata(
    query: str,
    nodes: list[NodeWithScore],
    user_profile: dict[str, Any],
    top_k: int = 5,
) -> list[NodeWithScore]:
    """
    Rerank retrieved nodes using cross-encoder relevance (70%) + metadata overlap (30%).

    Args:
        query: Original user query
        nodes: Retrieved nodes from hybrid search (typically 15)
        user_profile: User's health profile dict with metadata fields
        top_k: Number of top results to keep after reranking

    Returns:
        Top-k nodes sorted by blended score (highest first)
    """
    if not nodes:
        return nodes

    logger.info(
        f"Metadata-weighted reranking: {len(nodes)} nodes, "
        f"user profile fields: {list(user_profile.keys())}"
    )

    reranker = get_reranker()

    # Step 1: Get cross-encoder scores
    pairs = [(query, node.get_content()) for node in nodes]
    cross_encoder_scores = reranker.predict(pairs)
    logger.debug(
        f"Raw cross-encoder scores: min={min(cross_encoder_scores):.3f}, max={max(cross_encoder_scores):.3f}"
    )

    # Step 2: Normalize cross-encoder scores to 0-1
    min_score = min(cross_encoder_scores)
    max_score = max(cross_encoder_scores)
    score_range = max_score - min_score if max_score != min_score else 1.0
    normalized_ce_scores = [(s - min_score) / score_range for s in cross_encoder_scores]
    logger.debug(
        f"Normalized cross-encoder scores: min={min(normalized_ce_scores):.3f}, max={max(normalized_ce_scores):.3f}"
    )

    # Step 3: Compute metadata overlap scores
    metadata_scores = [
        compute_metadata_overlap_score(user_profile, node.node.metadata) for node in nodes
    ]
    logger.debug(
        f"Metadata overlap scores: min={min(metadata_scores):.3f}, max={max(metadata_scores):.3f}"
    )

    # Step 4: Blend (70% cross-encoder + 30% metadata)
    for node, ce_score, meta_score in zip(
        nodes, normalized_ce_scores, metadata_scores, strict=False
    ):
        node.score = float((0.7 * ce_score) + (0.3 * meta_score))

    # Step 5: Sort and keep top-k
    reranked = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[:top_k]

    logger.info(
        f"Reranked {len(nodes)} → {len(reranked)} nodes. "
        f"Top score: {reranked[0].score:.3f}, Bottom score: {reranked[-1].score:.3f}"
    )

    return reranked
