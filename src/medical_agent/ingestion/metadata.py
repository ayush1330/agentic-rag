"""
Document-level medical metadata extraction.

Extracts metadata from the paper's title + abstract (not from individual chunks),
then stamps the result onto all chunks. This ensures each paper gets unique,
accurate metadata based on its actual content rather than similar chunk samples.

Key design:
- Extract from title + abstract (highest signal, most unique per paper)
- Fallback chain: abstract → introduction → first 3 chunks
- Light text cleaning for Docling unicode artifacts
- No cache needed (one extraction per paper from unique document text)
"""

import logging
import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.llms.azure_openai import AzureOpenAI

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


class MedicalMetadata(BaseModel):
    """Medical metadata extracted from research papers."""

    ethnicities: list[str] = Field(
        default_factory=list,
        description=(
            "Ethnicity terms EXPLICITLY mentioned. Must be one of: "
            "African / Black, Asian, Caucasian, Hispanic / Latina, Middle Eastern, "
            "Mixed, Native American / Indigenous, North African, Pacific Islander, "
            "South Asian, Southeast Asian. Only extract if explicitly stated."
        ),
    )

    diagnoses: list[str] = Field(
        default_factory=list,
        description=(
            "Hormone-related diagnoses EXPLICITLY mentioned. Must be one of: "
            "Adenomyosis, Endometriosis, Bacterial vaginosis, Yeast infection, "
            "Sexually transmitted infection, Polycystic ovary syndrome (PCOS), "
            "Premature ovarian insufficiency, Thyroid disorder, Fibroids (uterine myomas), "
            "Ovarian cysts, Pelvic inflammatory disease. Only extract if explicitly stated."
        ),
    )

    symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "Symptoms EXPLICITLY mentioned. Must be one of: "
            "Creamy, Clear, Yellow, Green, Gray, Pink, Brown (discharge colors), "
            "Vaginal Odor, Itchy, Burning, Pelvic Pain, Swelling, Redness, "
            "Vaginal Dryness, Frequent Urination, Painful Urination. "
            "Only extract if explicitly stated."
        ),
    )

    menstrual_status: list[str] = Field(
        default_factory=list,
        description=(
            "Menstrual cycle states EXPLICITLY mentioned. Must be one of: "
            "Premenstrual, Menstrual, Postmenstrual, Ovulation, Luteal Phase, "
            "Follicular Phase. Only extract if explicitly stated."
        ),
    )

    birth_control: list[str] = Field(
        default_factory=list,
        description=(
            "Birth control types EXPLICITLY mentioned. Must be one of: "
            "Pill, IUD, Implant, Patch, Ring, Injection, Condom, Diaphragm, "
            "Sterilization. Only extract if explicitly stated."
        ),
    )

    hormone_therapy: list[str] = Field(
        default_factory=list,
        description=(
            "Hormone therapies EXPLICITLY mentioned. Must be one of: "
            "HRT, Testosterone, Progesterone, Estrogen, Thyroid Medication. "
            "Only extract if explicitly stated."
        ),
    )

    fertility_treatments: list[str] = Field(
        default_factory=list,
        description=(
            "Fertility treatments EXPLICITLY mentioned. Must be one of: "
            "IVF, IUI, Clomiphene, Letrozole, Gonadotropins, Ovulation Induction. "
            "Only extract if explicitly stated."
        ),
    )

    age_mentioned: bool = Field(
        default=False, description="Whether age or age range is explicitly mentioned in the text"
    )

    age_range: str | None = Field(
        default=None,
        description="Age range if mentioned (e.g., '25-35', '18-45'). None if not mentioned.",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for extraction quality (0.0-1.0). Higher is better.",
    )


EXTRACTION_PROMPT = """You are a medical metadata extractor for a women's health RAG system. Your output is stamped onto all chunks of each paper and used for filtering (e.g., "show me papers about PCOS" or "bacterial vaginosis treatment"). Accuracy is critical—wrong metadata causes failed filters and poor retrieval.

**Paper title:** {title}

Use the paper title as your primary guide: it often states the main condition(s) studied. Cross-check with the text below.

TASK:
Extract ONLY information that is EXPLICITLY stated in the text. Output structured metadata for filtering.

STRICT RULES (violations cause bad retrieval):
1. EXTRACT ONLY if the term or concept is directly mentioned—no inference, extrapolation, or assumption
2. If the paper mentions "vaginal pH in BV patients", extract "Bacterial vaginosis" and "vaginal pH"—the text explicitly references both
3. DO NOT extract based on: related conditions, implications, or "papers about X typically also discuss Y"
4. DO NOT extract ethnicity from study location alone (e.g., "study in Japan" ≠ Asian)
5. Use EXACT standardized terms from the lists below—no synonyms or paraphrases in output

VOCABULARY (use these exact strings):
- ethnicities: African/Black, Asian, Caucasian, Hispanic/Latina, Middle Eastern, Mixed, Native American/Indigenous, North African, Pacific Islander, South Asian, Southeast Asian
- diagnoses: Adenomyosis, Endometriosis, Bacterial vaginosis, Yeast infection, Sexually transmitted infection, Polycystic ovary syndrome (PCOS), Premature ovarian insufficiency, Thyroid disorder, Fibroids (uterine myomas), Ovarian cysts, Pelvic inflammatory disease
- symptoms: Creamy, Clear, Yellow, Green, Gray, Pink, Brown, Vaginal Odor, Itchy, Burning, Pelvic Pain, Swelling, Redness, Vaginal Dryness, Frequent Urination, Painful Urination
- birth_control: Pill, IUD, Implant, Patch, Ring, Injection, Condom, Diaphragm, Sterilization
- hormone_therapy: HRT, Testosterone, Progesterone, Estrogen, Thyroid Medication
- fertility_treatments: IVF, IUI, Clomiphene, Letrozole, Gonadotropins, Ovulation Induction
- menstrual_status: Premenstrual, Menstrual, Postmenstrual, Ovulation, Luteal Phase, Follicular Phase
- age_range: e.g. "25-35", "18-45" (only if explicitly stated)

CANONICAL MAPPINGS (map variants to standard terms):
- candida, candidal, C. albicans → Yeast infection
- BV, bacterial vaginosis → Bacterial vaginosis
- birth control pill, oral contraceptive, OCP → Pill
- hormone replacement therapy, HRT, estrogen therapy → HRT
- STI, STD, sexually transmitted → Sexually transmitted infection

OUTPUT:
- Empty list [] for any category with no explicit mention
- confidence: 0.0-1.0 — 0.8+ when multiple clear mentions, 0.5-0.7 when few, 0.2-0.4 when very sparse, 0.0 when nothing extractable

Extract from the following text:

{text}"""


# Docling unicode artifacts to clean
_UNICODE_FIXES = [
    ("\ufb01", "fi"),  # fi ligature
    ("\ufb02", "fl"),  # fl ligature
    ("\u2019", "'"),  # right single quote
    ("\u2018", "'"),  # left single quote
    ("\u201c", '"'),  # left double quote
    ("\u201d", '"'),  # right double quote
    ("\u2013", "-"),  # en dash
    ("\u2014", "-"),  # em dash
]

_CITATION_RE = re.compile(r"\[\d+(?:[-,]\s*\d+)*\]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """Light cleaning for Docling unicode artifacts and citation markers."""
    for old, new in _UNICODE_FIXES:
        text = text.replace(old, new)
    text = _CITATION_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _find_section_text(
    nodes: Sequence[BaseNode],
    section_keywords: list[str],
    max_chars: int = 3000,
) -> str | None:
    """Find text from a specific section using Docling headings metadata."""
    parts = []
    total = 0

    for node in nodes:
        headings = node.metadata.get("headings") or []
        heading_text = " > ".join(headings).lower() if headings else ""

        if any(kw in heading_text for kw in section_keywords):
            content = node.get_content()
            parts.append(content)
            total += len(content)
            if total >= max_chars:
                break

    if not parts:
        return None

    return "\n\n".join(parts)[:max_chars]


def _build_extraction_text(
    title: str | None,
    nodes: Sequence[BaseNode],
) -> str:
    """
    Build the best text for metadata extraction with fallback chain.

    Priority: abstract → introduction → first 3 raw chunks.
    """
    parts = []

    # 1. Try abstract
    abstract = _find_section_text(nodes, ["abstract"])
    if abstract:
        parts.append(f"ABSTRACT:\n{abstract}")

    # 2. Try methods/results for additional context
    methods = _find_section_text(nodes, ["method", "material"], max_chars=2000)
    if methods:
        parts.append(f"METHODS:\n{methods}")

    results = _find_section_text(nodes, ["result"], max_chars=2000)
    if results:
        parts.append(f"RESULTS:\n{results}")

    # 3. If no abstract found, try introduction as fallback
    if not abstract:
        intro = _find_section_text(nodes, ["introduction", "background"], max_chars=2000)
        if intro:
            parts.append(f"INTRODUCTION:\n{intro}")

    # 4. Final fallback: first 3 raw chunks
    if not parts:
        logger.warning("No section headings found — falling back to first 3 chunks")
        for node in nodes[:3]:
            parts.append(node.get_content()[:2000])

    text = "\n\n".join(parts)

    # Clean Docling artifacts
    text = _clean_text(text)

    return text


def _get_llm() -> LLM:
    """Create the Azure OpenAI LLM for metadata extraction."""
    model = getattr(
        settings,
        "metadata_extraction_model",
        settings.azure_openai_deployment_name,
    )
    return AzureOpenAI(
        model=model,
        deployment_name=model,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )


def _empty_metadata() -> dict[str, Any]:
    """Return empty metadata structure."""
    return {
        "ethnicities": [],
        "diagnoses": [],
        "symptoms": [],
        "menstrual_status": [],
        "birth_control": [],
        "hormone_therapy": [],
        "fertility_treatments": [],
        "age_mentioned": False,
        "age_range": None,
        "confidence": 0.0,
    }


async def extract_medical_metadata(
    title: str | None,
    nodes: Sequence[BaseNode],
    llm: LLM | None = None,
) -> dict[str, Any]:
    """
    Extract medical metadata from a paper's content.

    Uses title + abstract (with fallback chain) for extraction,
    producing unique metadata per paper.

    Args:
        title: Paper title from Docling
        nodes: Raw chunk nodes (before filtering/transformation)
        llm: Optional LLM instance (creates one if not provided)

    Returns:
        Dictionary of extracted medical metadata
    """
    if not nodes:
        logger.warning("No nodes provided for metadata extraction")
        return _empty_metadata()

    if not getattr(settings, "metadata_extraction_enabled", True):
        logger.warning("Metadata extraction is disabled in settings")
        return _empty_metadata()

    if llm is None:
        llm = _get_llm()

    # Build extraction text from document sections
    text = _build_extraction_text(title, nodes)
    paper_id = nodes[0].metadata.get("paper_id", "unknown")

    if not text.strip():
        logger.warning(f"No text available for extraction (paper {paper_id})")
        return _empty_metadata()

    logger.info(
        f"Extracting metadata for paper '{title or 'Unknown'}' "
        f"(id: {paper_id}, text: {len(text)} chars)"
    )

    try:
        prompt = ChatPromptTemplate.from_messages([("user", EXTRACTION_PROMPT)])

        result: MedicalMetadata = await llm.astructured_predict(
            output_cls=MedicalMetadata,
            prompt=prompt,
            title=title or "Unknown",
            text=text,
        )

        metadata: dict[str, Any] = result.model_dump()

        total_terms = sum(len(v) for v in metadata.values() if isinstance(v, list))
        logger.info(
            f"Extracted {total_terms} terms for paper '{title}', "
            f"confidence: {metadata.get('confidence', 0.0):.2f}"
        )

        return metadata

    except Exception as e:
        logger.error(f"Metadata extraction failed for paper {paper_id}: {e}", exc_info=True)
        return _empty_metadata()


def stamp_metadata_on_nodes(
    nodes: Sequence[BaseNode],
    metadata: dict[str, Any],
) -> list[BaseNode]:
    """
    Stamp extracted metadata onto all chunk nodes.

    Args:
        nodes: Processed chunk nodes
        metadata: Extracted medical metadata dict

    Returns:
        Same nodes with metadata added (modified in-place and returned)
    """
    for node in nodes:
        if node.metadata is None:
            node.metadata = {}
        node.metadata.update(metadata)

    logger.info(f"Stamped metadata on {len(nodes)} nodes")
    return list(nodes)
