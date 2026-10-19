"""
Query API Routes

Provides the main endpoint for pH analysis queries using the
medical reasoning agent.
"""

import json
import logging
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.agents import medical_rag_app
from medical_agent.agents.utils import real_selections
from medical_agent.api.schemas import (
    CitationResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)
from medical_agent.core.config import settings
from medical_agent.infrastructure.database.models import Paper, QueryLog
from medical_agent.infrastructure.database.session import get_async_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Query"])


# ============================================================================
# Citation Renumbering + Filtering Pass (DECIDE-04 PROMPT-03)
# ============================================================================


def renumber_and_filter_citations(
    your_ph: str,
    your_symptoms: str | None,
    your_personal_baseline: str | None,
    your_health_context: str | None,
    next_steps: str,
    overview: str,
    raw_citations: list[dict],
    paper_map: dict,
) -> tuple[str, str | None, str | None, str | None, str, str, list[CitationResponse]]:
    """Backend renumbering + filtering pass per DECIDE-04 PROMPT-03.

    Walks the 5 deep-dive fields in locked order (your_ph → your_symptoms →
    your_personal_baseline → your_health_context → next_steps), then `overview`
    LAST so it reuses numbers established by the deep dive.
    Builds a chunk_idx -> new_id remap with paper-level dedup (first-mention-of-paper
    wins; later chunks from the same paper reuse the existing new_id).
    Rewrites markers, drops hallucinated/orphan markers, and builds CitationResponse[]
    in 1..N order with URL fields rendered server-side.

    Args:
        your_ph: LLM's 'Your pH' section content (required, always populated).
        your_symptoms: LLM's 'Your symptoms' section content (None when absent).
        your_personal_baseline: LLM's 'Your personal baseline' section content (None when absent).
        your_health_context: LLM's 'Your health context' section content (None when absent).
        next_steps: LLM's 'Next steps' section content (required, always populated).
        overview: LLM's overview text (may be empty).
        raw_citations: list of dicts shaped {id: int, paper_id: str | None, score: float}
            from format_retrieved_nodes.
        paper_map: dict of paper_id (str) -> Paper ORM row (or FakePaper in tests).

    Returns:
        (new_your_ph, new_your_symptoms, new_your_personal_baseline,
         new_your_health_context, new_next_steps, new_overview, citations)
        - 5 deep-dive fields rewritten with renumbered markers (None preserved
          for absent optional fields).
        - new_overview: rewritten overview with renumbered markers (hallucinations dropped).
        - citations: list[CitationResponse] in 1..N order (one per unique paper).
    """
    chunk_to_paper: dict[int, str | None] = {
        c["id"]: c.get("paper_id") for c in raw_citations
    }
    remap: dict[int, int] = {}
    paper_to_new_id: dict[str, int] = {}
    counter = 1

    def _process_text(text: str) -> str:
        nonlocal counter

        def replace_marker(m: "re.Match[str]") -> str:
            nonlocal counter
            chunk_idx = int(m.group(1))
            paper_id = chunk_to_paper.get(chunk_idx)
            # Edge case #1 (hallucinated): chunk_idx not in raw_citations
            # Edge case #5 (orphan): paper_id has no row in paper_map
            if paper_id is None or paper_id not in paper_map:
                logger.warning(
                    "Dropping citation marker [%d] — chunk not in raw_citations or paper_id orphan",
                    chunk_idx,
                )
                return ""
            if chunk_idx in remap:
                return f"[{remap[chunk_idx]}]"
            if paper_id in paper_to_new_id:
                # Edge case #4: same paper, second chunk — reuse existing id
                new_id = paper_to_new_id[paper_id]
                remap[chunk_idx] = new_id
                return f"[{new_id}]"
            # First mention of this paper
            remap[chunk_idx] = counter
            paper_to_new_id[paper_id] = counter
            counter += 1
            return f"[{remap[chunk_idx]}]"

        def multi_replace(m: "re.Match[str]") -> str:
            raw = m.group(1)          # e.g. "1, 5" or "1"
            parts = [p.strip() for p in raw.split(",")]
            results = []
            for part in parts:
                # Build a fake single-digit match object and delegate to replace_marker.
                # Use re.match on a synthesized string so replace_marker receives a real
                # re.Match with group(1) = part.
                fake_m = re.match(r"(\d+)", part)
                if fake_m is not None:
                    resolved = replace_marker(fake_m)
                    if resolved:   # drop empty-string results (hallucinated/orphan)
                        results.append(resolved)
            return " ".join(results)

        return re.sub(r"\[(\d+(?:,\s*\d+)*)\]", multi_replace, text)

    def _process_optional(text: str | None) -> str | None:
        if text is None:
            return None
        return _process_text(text)

    # Pitfall 7 (updated): deep-dive sections processed FIRST in locked walk
    # order; overview LAST so it reuses numbers established by the deep dive.
    new_your_ph = _process_text(your_ph)
    new_your_symptoms = _process_optional(your_symptoms)
    new_your_personal_baseline = _process_optional(your_personal_baseline)
    new_your_health_context = _process_optional(your_health_context)
    new_next_steps = _process_text(next_steps)
    new_overview = _process_text(overview)

    # Build CitationResponse[] in new_id order
    new_id_to_paper_id = {v: k for k, v in paper_to_new_id.items()}
    result: list[CitationResponse] = []
    for new_id in sorted(new_id_to_paper_id.keys()):
        pid = new_id_to_paper_id[new_id]
        p = paper_map[pid]
        result.append(
            CitationResponse(
                id=new_id,
                title=p.title,
                nlm_reference=p.nlm_reference,
                pmid_url=f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/" if p.pmid else None,
                pmcid_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{p.pmcid}/" if p.pmcid else None,
                doi_url=f"https://doi.org/{p.doi}" if p.doi else None,
            )
        )

    return (
        new_your_ph,
        new_your_symptoms,
        new_your_personal_baseline,
        new_your_health_context,
        new_next_steps,
        new_overview,
        result,
    )


# ============================================================================
# Citation Mapping Helper (async — Paper join + renumbering pass)
# ============================================================================


async def build_citation_responses(
    raw_citations: list[dict],
    your_ph: str,
    your_symptoms: str | None,
    your_personal_baseline: str | None,
    your_health_context: str | None,
    next_steps: str,
    overview: str,
    session: AsyncSession,
) -> tuple[str, str | None, str | None, str | None, str, str, list[CitationResponse]]:
    """Async route-layer function: batched Paper join + renumbering pass.

    Implements DECIDE-03 D-05 (Paper join at route layer), API-06 (server-side URL
    rendering), API-07 (chunk-to-paper dedup), API-08 (numbering authority).

    Args:
        raw_citations: from format_retrieved_nodes — shape {id, paper_id, score}.
        your_ph: LLM's 'Your pH' section (from MedicalResponse.your_ph).
        your_symptoms: LLM's 'Your symptoms' section (None when absent).
        your_personal_baseline: LLM's 'Your personal baseline' section (None when absent).
        your_health_context: LLM's 'Your health context' section (None when absent).
        next_steps: LLM's 'Next steps' section (from MedicalResponse.next_steps).
        overview: LLM's overview text (from MedicalResponse.overview).
        session: AsyncSession from FastAPI Depends(get_async_session).

    Returns:
        (new_your_ph, new_your_symptoms, new_your_personal_baseline,
         new_your_health_context, new_next_steps, new_overview, citations) —
        the 5 deep-dive fields + overview rewritten by the renumbering pass.
    """
    # Early return: empty raw_citations means no Paper query and no markers to renumber
    if not raw_citations:
        # Still run the regex pass so any hallucinated markers in text are dropped
        return renumber_and_filter_citations(
            your_ph=your_ph,
            your_symptoms=your_symptoms,
            your_personal_baseline=your_personal_baseline,
            your_health_context=your_health_context,
            next_steps=next_steps,
            overview=overview,
            raw_citations=[],
            paper_map={},
        )

    # Pitfall 1: paper_id stored as UUID string in metadata; cast to uuid.UUID for query
    paper_id_strs = [c["paper_id"] for c in raw_citations if c.get("paper_id")]
    if not paper_id_strs:
        # All chunks lack paper_id — orphans; renumber pass drops everything
        return renumber_and_filter_citations(
            your_ph=your_ph,
            your_symptoms=your_symptoms,
            your_personal_baseline=your_personal_baseline,
            your_health_context=your_health_context,
            next_steps=next_steps,
            overview=overview,
            raw_citations=raw_citations,
            paper_map={},
        )

    try:
        paper_uuids = list({uuid.UUID(pid) for pid in paper_id_strs})
    except ValueError as e:
        logger.error("Invalid UUID in paper_id metadata: %s", e)
        raise

    # D-06: ONE round-trip — batched in_() query, select only columns needed for CitationResponse
    # select(Paper) would pull all ORM-mapped columns including ones not yet migrated in dev DB
    result = await session.execute(
        select(Paper.id, Paper.title, Paper.nlm_reference, Paper.pmid, Paper.pmcid, Paper.doi)
        .where(Paper.id.in_(paper_uuids))
    )
    papers = result.all()
    paper_map = {str(p.id): p for p in papers}

    return renumber_and_filter_citations(
        your_ph=your_ph,
        your_symptoms=your_symptoms,
        your_personal_baseline=your_personal_baseline,
        your_health_context=your_health_context,
        next_steps=next_steps,
        overview=overview,
        raw_citations=raw_citations,
        paper_map=paper_map,
    )


# ============================================================================
# Query Generation and Rewriting Helpers
# ============================================================================


async def generate_initial_query(
    ph_value: float,
    health_profile: dict[str, Any],
    symptoms: list[str],
) -> tuple[str, str]:
    """
    Generate dual queries from form data: one for retrieval, one for response generation.

    Produces two complementary queries in a single LLM call:
    - retrieval_query: clinical, keyword-rich, optimized for BM25 + vector search
    - display_query: first-person, conversational, used as {query} in generation prompt

    Args:
        ph_value: Vaginal pH reading
        health_profile: User's health context (diagnoses, age, etc.)
        symptoms: List of reported symptoms

    Returns:
        Tuple of (retrieval_query, display_query)
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    diagnoses_str = ", ".join(real_selections(health_profile.get("diagnoses")))
    symptoms_str = ", ".join(symptoms[:3]) if symptoms else "none reported"
    age_str = f"{health_profile.get('age')} years old" if health_profile.get("age") else "adult"

    prompt = f"""You generate two queries from a patient's health data. These queries search a database of peer-reviewed women's vaginal health research papers.

<patient_context>
- Vaginal pH: {ph_value}
- Age: {age_str}
- Diagnoses: {diagnoses_str if diagnoses_str else "none"}
- Symptoms: {symptoms_str}
</patient_context>

Generate TWO queries as a JSON object:

1. **retrieval_query**: A clinical, keyword-rich medical research query optimized for literature search. Use precise medical terminology. Prioritize the most clinically significant factor — if diagnoses are present, lead with them; if only symptoms, lead with the symptom-pH correlation. Do NOT include phrases like "research about" or "studies on".

2. **display_query**: A first-person, conversational question as if the patient is asking directly. Use "my" and "I". Keep it natural and simple.

<example>
Input: pH 5.2, 54 years old, Bacterial vaginosis, itching
Output:
{{"retrieval_query": "Clinical significance of elevated vaginal pH above 5.0 in bacterial vaginosis with vulvar pruritus in perimenopausal women", "display_query": "What does my pH of 5.2 mean given my BV diagnosis, and should I be concerned about the itching?"}}
</example>

Output ONLY a valid JSON object with "retrieval_query" and "display_query" keys:"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    response_text = str(response.content).strip()

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            logger.warning("Could not parse dual-query JSON, falling back to single query")
            return response_text, response_text

    retrieval_query = result.get("retrieval_query", response_text)
    display_query = result.get("display_query", retrieval_query)

    logger.info(f"Retrieval query: {retrieval_query[:100]}...")
    logger.info(f"Display query: {display_query[:100]}...")
    return retrieval_query, display_query


async def rewrite_followup_query(
    user_message: str,
    chat_history: list,
    ph_value: float,
    health_profile: dict[str, Any],
) -> str:
    """
    Rewrite follow-up query with conversation history context.

    Transforms context-dependent follow-up questions into standalone search queries
    by incorporating relevant information from conversation history and user profile.
    These queries search a database of peer-reviewed women's vaginal health research papers.

    Args:
        user_message: Current user question
        chat_history: Previous conversation messages
        ph_value: User's pH value
        health_profile: User's health context

    Returns:
        Rewritten standalone search query
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    history_text = "\n".join(
        [
            f"{msg.type if hasattr(msg, 'type') else 'message'}: {str(msg.content if hasattr(msg, 'content') else msg)[:300]}"
            for msg in chat_history[-4:]
        ]
    )

    diagnoses_str = ", ".join(real_selections(health_profile.get("diagnoses")))

    # symptoms is stored as a dict-of-lists (discharge/vulva_vagina/smell/urine)
    # in the new schema. Flatten for this retrieval prompt.
    symptoms_value = health_profile.get("symptoms", [])
    if isinstance(symptoms_value, dict):
        symptoms_flat = [s for group in symptoms_value.values() for s in group]
    else:
        symptoms_flat = symptoms_value or []
    symptoms_str = ", ".join(real_selections(symptoms_flat))

    prompt = f"""Rewrite the current follow-up question as a standalone search query for a database of peer-reviewed women's vaginal health research papers.

<conversation_history>
{history_text}
</conversation_history>

<user_profile>
- pH: {ph_value}
- Diagnoses: {diagnoses_str if diagnoses_str else "none"}
- Symptoms: {symptoms_str if symptoms_str else "none reported"}
</user_profile>

Current Question: "{user_message}"

<instructions>
- Include relevant context from the conversation and profile ONLY if it helps clarify the question
- Use precise medical terminology for better retrieval
- If the question is already clear and complete, return it as-is
- Do NOT add "research about" or "studies on" — write it as a direct clinical question
</instructions>

<example>
History: user asked about high pH and BV, assistant explained treatment options
Follow-up: "Are there home remedies for this?"
Rewritten: "Non-prescription home remedies and natural treatments for bacterial vaginosis with elevated vaginal pH"
</example>

Rewritten Query:"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    rewritten = str(response.content).strip()

    logger.info(f"Rewritten query: '{user_message}' → '{rewritten[:100]}...'")
    return rewritten


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Analyze pH reading",
    description="""
    Submit a pH reading for analysis using the medical RAG agent.

    The agent will:
    1. Analyze the pH value and any symptoms
    2. Search relevant medical research papers
    3. Assess the risk level
    4. Generate evidence-based insights with citations

    **Important**: This is purely informational and NOT medical advice.
    Always consult a healthcare provider for medical concerns.
    """,
)
async def analyze_ph(
    request: QueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> QueryResponse:
    """
    Analyze a pH reading and return evidence-based insights.

    Args:
        request: The query request with pH value and optional symptoms

    Returns:
        Analysis response with risk level, insights, and citations
    """
    # Verify Azure OpenAI is configured
    if not settings.is_azure_openai_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "SERVICE_NOT_CONFIGURED",
                "message": "Azure OpenAI is not configured. LLM features are unavailable.",
            },
        )

    logger.info(f"Processing query: pH={request.ph_value}")

    start_time = time.time()

    try:
        # ── Build health profile from request ──
        # Raw selections are preserved here (including UI sentinels like
        # "None of these" / "Prefer not to say"); build_health_context
        # translates them into clean prompt text ("None reported" /
        # "Not disclosed"). The retrieval-query path uses real_selections()
        # to strip sentinels, since they are noise for search.
        health_profile: dict[str, Any] = {}

        # Step 1: About you
        if request.age is not None:
            health_profile["age"] = request.age
        if request.life_stage:
            health_profile["life_stage"] = request.life_stage
        if request.ethnic_backgrounds:
            health_profile["ethnic_backgrounds"] = request.ethnic_backgrounds
        if request.ethnic_background_other:
            health_profile["ethnic_background_other"] = request.ethnic_background_other

        # Step 2: Cycle, hormones & medications
        if request.menstrual_cycle:
            health_profile["menstrual_cycle"] = request.menstrual_cycle
        if request.medications:
            health_profile["medications"] = request.medications
        if request.diagnoses:
            health_profile["diagnoses"] = request.diagnoses

        # Birth control sub-form: preserve the structure (recent_changes / pill
        # / iud / other_methods / permanent) so build_health_context can emit
        # labeled lines for each sub-field.
        if request.birth_control:
            bc = request.birth_control
            bc_block: dict[str, Any] = {}
            if bc.recent_changes:
                bc_block["recent_changes"] = bc.recent_changes
            if bc.pill:
                bc_block["pill"] = bc.pill
            if bc.iud:
                bc_block["iud"] = bc.iud
            if bc.other_methods:
                bc_block["other_methods"] = bc.other_methods
            if bc.permanent:
                bc_block["permanent"] = bc.permanent
            if bc_block:
                health_profile["birth_control"] = bc_block

        # Fertility journey sub-form
        if (
            request.fertility_journey
            and request.fertility_journey.fertility_treatments
        ):
            health_profile["fertility_treatments"] = (
                request.fertility_journey.fertility_treatments
            )

        # Step 3: Symptoms — preserve sub-group labels (discharge / vulva_vagina
        # / smell / urine) AND the raw selections (sentinels included) for the
        # prompt; build_health_context translates the sentinels. flat_symptoms
        # is a sentinel-free list used only by the retrieval-query generator.
        symptoms_grouped: dict[str, list[str]] = {}
        flat_symptoms: list[str] = []
        if request.symptoms:
            for group, values in (
                ("discharge", request.symptoms.discharge or []),
                ("vulva_vagina", request.symptoms.vulva_vagina or []),
                ("smell", request.symptoms.smell or []),
                ("urine", request.symptoms.urine or []),
            ):
                if values:
                    symptoms_grouped[group] = values
                    flat_symptoms.extend(real_selections(values))
        if symptoms_grouped:
            health_profile["symptoms"] = symptoms_grouped

        # Last step: Recent vaginal context (raw selections preserved for the
        # prompt; build_health_context translates sentinels).
        if request.recent_vaginal_context:
            rvc = request.recent_vaginal_context
            rvc_block: dict[str, list[str]] = {}
            if rvc.vaginal_products:
                rvc_block["vaginal_products"] = rvc.vaginal_products
            if rvc.sex_fluids:
                rvc_block["sex_fluids"] = rvc.sex_fluids
            if rvc.menstrual_blood_or_spotting:
                rvc_block["menstrual_blood_or_spotting"] = (
                    rvc.menstrual_blood_or_spotting
                )
            if rvc_block:
                health_profile["recent_vaginal_context"] = rvc_block

        if request.symptoms and request.symptoms.notes:
            health_profile["notes"] = request.symptoms.notes

        logger.info(
            f"Health profile: age={health_profile.get('age')}, "
            f"life_stage={health_profile.get('life_stage')}, "
            f"symptoms={len(flat_symptoms)}"
        )

        # Use provided session_id or generate new one
        # Reusing thread_id is what gives LangGraph memory continuity
        # across follow-up API calls.
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        logger.info(f"Session ID: {session_id}")

        # Generate or rewrite query based on interaction type
        display_query = ""
        if request.user_message:
            # FOLLOW-UP: Load conversation history and rewrite query
            logger.info("Follow-up question detected - loading conversation history")

            try:
                state_snapshot = medical_rag_app.get_state(
                    config={"configurable": {"thread_id": session_id}}
                )
                chat_history = state_snapshot.values.get("messages", [])

                logger.info(f"Loaded {len(chat_history)} messages from conversation history")

                query_text = await rewrite_followup_query(
                    user_message=request.user_message,
                    chat_history=chat_history,
                    ph_value=request.ph_value,
                    health_profile=health_profile,
                )
                # Follow-ups use the user's own words — no display_query needed
                display_query = request.user_message

            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}. Using raw query.")
                query_text = request.user_message
                display_query = request.user_message

        else:
            # INITIAL: Generate dual queries from form data
            logger.info("Initial interaction - generating dual queries from form data")

            retrieval_query, display_query = await generate_initial_query(
                ph_value=request.ph_value,
                health_profile=health_profile,
                symptoms=flat_symptoms,
            )
            query_text = retrieval_query

        # Prepare state for LangGraph
        initial_state = {
            "messages": [{"role": "user", "content": query_text}],
            "ph_value": request.ph_value,
            "health_profile": health_profile if health_profile else {},
            "display_query": display_query,
            "retry_count": 0,
            "refinement_history": [],
            "skip_retry": False,
        }

        # Run LangGraph workflow with session continuity (async)
        logger.info(f"Invoking medical RAG graph for session: {session_id}")

        graph_result = await medical_rag_app.ainvoke(
            initial_state, config={"configurable": {"thread_id": session_id}}
        )

        # Extract response from graph output payload — read the 5 deep-dive
        # fields directly from state (populated by generate_node). The
        # `messages` list still carries a concat of the 5 fields for any
        # downstream LangGraph node, but the route layer does not read it.
        raw_citations = graph_result.get("citations", [])
        overview_raw = graph_result.get("overview", "")
        your_ph_raw = graph_result.get("your_ph", "") or ""
        your_symptoms_raw = graph_result.get("your_symptoms")  # may be None
        your_personal_baseline_raw = graph_result.get("your_personal_baseline")  # may be None
        your_health_context_raw = graph_result.get("your_health_context")  # may be None
        next_steps_raw = graph_result.get("next_steps", "") or ""

        (
            new_your_ph,
            new_your_symptoms,
            new_your_personal_baseline,
            new_your_health_context,
            new_next_steps,
            new_overview,
            citations,
        ) = await build_citation_responses(
            raw_citations=raw_citations,
            your_ph=your_ph_raw,
            your_symptoms=your_symptoms_raw,
            your_personal_baseline=your_personal_baseline_raw,
            your_health_context=your_health_context_raw,
            next_steps=next_steps_raw,
            overview=overview_raw,
            session=db,
        )

        logger.info(
            f"Analysis complete: {len(citations)} unique citations (of {len(raw_citations)} retrieved chunks)"
        )

        # Build medical disclaimer
        disclaimers = (
            "This analysis is for informational purposes only and does not constitute medical advice. "
            "Always consult with a qualified healthcare provider for medical concerns, diagnosis, or treatment."
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # QueryLog.response is a single TEXT column — concat the 5 renumbered
        # deep-dive fields (skip None / empty) so beta-mode logs stay queryable
        # as a single blob. DB schema unchanged.
        deep_dive_concat = "\n\n".join(
            s
            for s in (
                new_your_ph,
                new_your_symptoms,
                new_your_personal_baseline,
                new_your_health_context,
                new_next_steps,
            )
            if s
        )

        # E-REQ-5: persist query_log row before responding (D-09 — beta mode anonymous)
        db.add(
            QueryLog(
                ph_value=request.ph_value,
                query_text=(getattr(request, "user_message", None) or display_query),
                health_profile_snapshot=health_profile,
                response=deep_dive_concat,
                citations={"citations": [c.model_dump() for c in citations]},
                retrieved_chunk_ids=[c["id"] for c in raw_citations],
                processing_time_ms=processing_time_ms,
                user_id=None,  # beta: anonymous
            )
        )
        await db.flush()  # session.py commits on context exit; do NOT call db.commit() here

        return QueryResponse(
            session_id=session_id,
            ph_value=request.ph_value,
            your_ph=new_your_ph,
            your_symptoms=new_your_symptoms,
            your_personal_baseline=new_your_personal_baseline,
            your_health_context=new_your_health_context,
            next_steps=new_next_steps,
            overview=new_overview,
            disclaimers=disclaimers,
            citations=citations,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PROCESSING_ERROR",
                "message": f"Failed to process query: {str(e)}",
            },
        ) from e
