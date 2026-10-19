"""
Contextual chunking: prepend document context to each chunk.

Based on Anthropic's contextual retrieval research (2024):
- Generates a short context for each chunk using a document outline
- Prepends context to chunk text before embedding
- Reduces retrieval errors by 49% when combined with hybrid search

Instead of passing raw first 8000 chars (which may only cover the intro),
we build an outline from title + abstract + section headings. This gives
the LLM a "map" of the full paper with fewer tokens and better coverage.
"""

import logging

from langchain_openai import AzureChatOpenAI
from llama_index.core.schema import BaseNode, TextNode

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """You are generating retrieval-optimized context for a women's health medical RAG system. This context is prepended to each chunk and used for both semantic (embedding) and keyword (BM25) search. Your goal: help this chunk match user questions about diagnoses, symptoms, treatments, and clinical findings.

<document_outline>
{doc_outline}
</document_outline>

<chunk>
{chunk_text}
</chunk>

Write 2-3 sentences (50-100 tokens) that maximize retrieval. The context will be prepended to the chunk before embedding and indexing.

REQUIREMENTS:
1. Open with: paper title + section (e.g., "From [Title]: In the Methods/Results section on...")
2. State what question this chunk helps answer (query-mirroring): e.g., "This chunk explains treatment efficacy for endometriosis" or "diagnostic methods for bacterial vaginosis"
3. Include exact medical terms users search for—use these canonical terms where relevant:
   - Diagnoses: bacterial vaginosis (BV), yeast infection, endometriosis, PCOS, pelvic inflammatory disease, fibroids, etc.
   - Symptoms: vaginal pH, discharge, odor, pelvic pain, etc.
   - Treatments: hormone therapy, HRT, antibiotics, etc.
   - Populations: premenopausal, menopausal, pregnant, age range
4. Front-load the most searchable terms in the first sentence
5. Be specific and factual—avoid generic phrases like "discusses various topics"

Output only the context. No preamble, explanations, or markdown."""


def _build_document_outline(
    nodes: list[BaseNode],
    title: str | None = None,
) -> str:
    """
    Build a compact document outline from title + abstract + section headings.

    Uses Docling's `headings` metadata from each node to reconstruct
    the paper's structure without needing the full raw text.
    """
    parts = []

    if title:
        parts.append(f"Title: {title}")

    # Extract abstract from first few nodes
    for node in nodes[:5]:
        headings = node.metadata.get("headings") or []
        heading_text = " > ".join(headings).lower() if headings else ""
        if "abstract" in heading_text:
            abstract = node.get_content()[:1500]
            parts.append(f"\nAbstract:\n{abstract}")
            break

    # Collect unique section headings in order
    seen_headings = []
    for node in nodes:
        headings = node.metadata.get("headings") or []
        if headings:
            heading_key = " > ".join(headings)
            if heading_key not in seen_headings:
                seen_headings.append(heading_key)

    if seen_headings:
        parts.append("\nSection outline:")
        for h in seen_headings:
            parts.append(f"- {h}")

    return "\n".join(parts) if parts else "No document outline available."


async def add_contextual_headers(
    nodes: list[BaseNode],
    title: str | None = None,
) -> list[BaseNode]:
    """
    Add document-level context to each chunk.

    Builds a document outline from Docling metadata (title + abstract +
    section headings), then calls gpt-4o-mini once per chunk to generate
    a 2-3 sentence context prefix.

    Args:
        nodes: Chunk nodes from parser
        title: Paper title from Docling

    Returns:
        Nodes with contextual headers prepended to text
    """
    if not nodes:
        return nodes

    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    doc_outline = _build_document_outline(nodes, title=title)
    logger.info(f"Document outline built ({len(doc_outline)} chars)")

    result_nodes = []
    success_count = 0
    failure_count = 0

    for idx, node in enumerate(nodes):
        chunk_text = node.get_content()

        # Skip very short chunks — not worth the LLM call
        if len(chunk_text.split()) < 20:
            result_nodes.append(node)
            continue

        try:
            response = await llm.ainvoke(
                CONTEXT_PROMPT.format(
                    doc_outline=doc_outline,
                    chunk_text=chunk_text[:1000],
                )
            )
            context = str(response.content).strip()

            contextualized_text = f"{context}\n\n{chunk_text}"

            new_node = TextNode(
                text=contextualized_text,
                metadata={
                    **node.metadata,
                    "contextual_header": context,
                    "has_context": True,
                },
                id_=node.node_id,
            )
            result_nodes.append(new_node)
            success_count += 1

            if (idx + 1) % 10 == 0:
                logger.info(f"Contextual headers: {idx + 1}/{len(nodes)} chunks processed")

        except Exception as e:
            logger.warning(f"Contextual header generation failed for chunk {idx + 1}: {e}")
            result_nodes.append(node)
            failure_count += 1

    logger.info(
        f"Contextual chunking complete: {success_count} contextualized, "
        f"{failure_count} failures, {len(nodes) - success_count - failure_count} skipped"
    )

    return result_nodes
