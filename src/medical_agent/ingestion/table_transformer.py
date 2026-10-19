"""
4
Convert table chunks to natural language for better embedding search.

Tables like "Treatment | Cure Rate | p-value" are hard to match via
vector search. Converting to natural language makes them semantically
searchable while preserving the original table in metadata.
"""

import logging

from langchain_openai import AzureChatOpenAI
from llama_index.core.schema import BaseNode, TextNode

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

TABLE_PROMPT = """Convert this medical research table into natural language optimized for RAG retrieval. The output replaces the table text and is used for both semantic (embedding) and keyword (BM25) search. Users ask questions like "What was the sensitivity of BV diagnosis?" or "treatment efficacy for endometriosis"—your sentences should match those query patterns.

**Table data:**
{table_text}

REQUIREMENTS:
1. Write 2-5 standalone factual sentences. Each sentence must be independently searchable—a user query could match any single sentence.
2. Phrase as direct answers to research questions, not meta-descriptions:
   - GOOD: "Metronidazole achieved 85% cure rate for bacterial vaginosis compared to 72% for placebo (p<0.05)."
   - BAD: "The table shows cure rates for different treatments."
3. Preserve ALL numbers, percentages, p-values, confidence intervals, and statistics EXACTLY as in the table. Do not round or approximate.
4. Include the condition/treatment/diagnosis in every sentence—use canonical medical terms: bacterial vaginosis (BV), yeast infection, endometriosis, PCOS, sensitivity, specificity, etc.
5. Make comparisons explicit: "higher than", "compared to", "associated with", "versus"
6. Front-load the most searchable terms (condition, outcome, key metric) in each sentence
7. Use proper medical terminology consistent with the paper's domain

Output only the natural language sentences. No preamble, bullets, or "Summary:" prefix."""


def _is_table_chunk(node: BaseNode) -> bool:
    """
    Detect table chunks using Docling's doc_items label metadata.

    Docling sets label="table" on table items in doc_items.
    """
    doc_items = node.metadata.get("doc_items") or []

    for item in doc_items:
        if isinstance(item, dict):
            label = (item.get("label") or "").lower()
        else:
            label = getattr(item, "label", "")
            if hasattr(label, "value"):
                label = label.value
            label = str(label).lower()

        if label == "table":
            return True

    return False


async def transform_table_chunks(nodes: list[BaseNode]) -> list[BaseNode]:
    """
    Find table chunks and convert them to natural language.

    Replaces table text with LLM-generated natural language while
    preserving original table data in metadata for reference.

    Args:
        nodes: All chunk nodes from parser

    Returns:
        Nodes with table chunks transformed to natural language
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    result_nodes = []
    table_count = 0

    for node in nodes:
        if not _is_table_chunk(node):
            result_nodes.append(node)
            continue

        table_text = node.get_content()

        # Skip empty or very short tables
        if not table_text.strip() or len(table_text.strip()) < 20:
            logger.warning("Skipping empty/short table chunk")
            result_nodes.append(node)
            continue

        try:
            logger.info(f"Transforming table chunk: {table_text[:80]}...")
            response = await llm.ainvoke(TABLE_PROMPT.format(table_text=table_text))
            natural_text = str(response.content).strip()

            new_node = TextNode(
                text=natural_text,
                metadata={
                    **node.metadata,
                    "original_table_text": table_text[:500],
                    "table_transformed": True,
                },
                id_=node.node_id,
            )
            result_nodes.append(new_node)
            table_count += 1
            logger.info(f"Table transformed: {natural_text[:100]}...")

        except Exception as e:
            logger.warning(f"Table transform failed, keeping original: {e}")
            result_nodes.append(node)

    if table_count > 0:
        logger.info(f"Table transform complete: {table_count} tables converted")

    return result_nodes
