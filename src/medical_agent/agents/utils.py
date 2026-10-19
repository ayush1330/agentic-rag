"""
Utility functions for RAG processing.

Handles formatting of retrieved nodes for citation-based responses.
"""

from typing import Any

from llama_index.core.schema import NodeWithScore


def format_retrieved_nodes(nodes: list[NodeWithScore]) -> tuple[str, list[dict[str, Any]]]:
    """
    Convert NodeWithScore objects into LLM prompt text and API citation dicts.

    Phase 4 TEARDOWN-02 shape:

    Each citation dict has the shape {id: int, paper_id: str | None, score: float}.
    The paper_id is the UUID string stamped by the ingestion pipeline.
    Resolution of citation metadata happens at the route layer via a Paper join.

    score is an INTERNAL field consumed by reasoning_node.compute_score_based_confidence
    and stripped by the route layer before reaching the API wire shape.
    """
    if not nodes:
        return "No relevant medical research documents found.", []

    docs_text_parts: list[str] = []
    citations: list[dict[str, Any]] = []

    for i, node_score in enumerate(nodes, 1):
        node = node_score.node
        chunk_text = node.get_content().strip()
        paper_id = node.metadata.get("paper_id")
        score = float(node_score.score) if node_score.score is not None else 0.0

        docs_text_parts.append(f"[{i}]: {chunk_text}")
        citations.append(
            {
                "id": i,
                "paper_id": paper_id,
                "score": score,
            }
        )

    return "\n\n".join(docs_text_parts), citations


# ── Sentinel handling ──────────────────────────────────────────────────────
# UI "no real answer" choices that must never reach the prompt or the retrieval
# query as if they were clinical data. Two kinds:
#   - affirmative "none": the user actively reported they have none
#   - withheld:           the user declined to answer
_AFFIRMATIVE_NONE = {"None", "None of these", "No discharge"}
_WITHHELD = {"Prefer not to say"}
_SENTINELS = _AFFIRMATIVE_NONE | _WITHHELD


def real_selections(values: list[str] | None) -> list[str]:
    """Return only genuine selections, dropping every sentinel.

    Used on the retrieval-query path, where sentinels are pure noise.
    """
    return [v for v in (values or []) if v not in _SENTINELS]


def render_selection(values: list[str] | str | None) -> str | None:
    """Turn a raw multi-select into clean prompt text, or None to omit the line.

    - real selections present   -> comma-joined (sentinels stripped)
    - only "Prefer not to say"  -> "Not disclosed"
    - only an affirmative none   -> "None reported"
    - empty / skipped           -> None
    """
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    if not values:
        return None
    real = real_selections(values)
    if real:
        return ", ".join(real)
    if any(v in _WITHHELD for v in values):
        return "Not disclosed"
    return "None reported"


def build_health_context(ph_value: float, health_profile: dict[str, Any]) -> str:
    """
    Build a section-grouped health context string from the user's profile.

    Fields are grouped under the result-section headers ([Your pH], [Symptoms],
    [Your personal baseline], [Health context]) so the prompt can enforce strict
    section separation — each Deep Dive field draws only from its matching group.
    Recent vaginal context (products / sex & fluids / spotting) lives under
    [Health context] per the v2 spec, NOT [Symptoms]. PCOS arrives under
    Menstrual Cycle and stays in [Your personal baseline]. Free-text notes are
    emitted last as unclassified context (not auto-treated as a symptom);
    classification is a later phase.

    Each line emits a clear `Label: value` pair. Sentinel choices are translated
    rather than passed through: an affirmative "none" (None / None of these /
    No discharge) becomes "None reported", "Prefer not to say" becomes "Not
    disclosed", and a skipped field is omitted entirely. See render_selection.

    Args:
        ph_value: Measured pH value
        health_profile: Dict produced by routes/query.py

    Returns:
        Section-grouped context string injected into <patient_context>.
    """
    # ── [Symptoms] ──
    symptom_lines: list[str] = []
    symptoms = health_profile.get("symptoms")
    if isinstance(symptoms, dict):
        for key, label in (
            ("discharge", "Discharge"),
            ("vulva_vagina", "Vulva & Vagina"),
            ("smell", "Smell"),
            ("urine", "Urine"),
        ):
            rendered = render_selection(symptoms.get(key))
            if rendered is not None:
                symptom_lines.append(f"{label}: {rendered}")
    elif isinstance(symptoms, list) and symptoms:
        symptom_lines.append(f"Reported: {', '.join(symptoms)}")

    # ── [Your personal baseline] ──
    baseline_lines: list[str] = []
    if age := health_profile.get("age"):
        baseline_lines.append(f"Age: {age}")
    # Life Stage — read by the prompt's CLINICAL-REVIEW rule to decide whether
    # to apply a life-stage label in the response.
    if (life_stage := render_selection(health_profile.get("life_stage"))) is not None:
        baseline_lines.append(f"Life Stage: {life_stage}")
    if (ethnicity := render_selection(health_profile.get("ethnic_backgrounds"))) is not None:
        baseline_lines.append(f"Ethnicity: {ethnicity}")
    if ethnic_other := health_profile.get("ethnic_background_other"):
        baseline_lines.append(f"Ethnicity (Other, free text): {ethnic_other}")
    if (menstrual_cycle := render_selection(health_profile.get("menstrual_cycle"))) is not None:
        baseline_lines.append(f"Menstrual Cycle: {menstrual_cycle}")

    # ── [Health context] — medications, diagnoses, birth control, fertility,
    # and recent vaginal context (per the v2 spec). ──
    health_context_lines: list[str] = []
    if (medications := render_selection(health_profile.get("medications"))) is not None:
        health_context_lines.append(f"Medications: {medications}")
    if (diagnoses := render_selection(health_profile.get("diagnoses"))) is not None:
        health_context_lines.append(f"Diagnoses: {diagnoses}")

    birth_control = health_profile.get("birth_control")
    if isinstance(birth_control, dict):
        if bc_recent := birth_control.get("recent_changes"):
            health_context_lines.append(
                f"Birth Control - Recent Changes: {', '.join(bc_recent)}"
            )
        if bc_pill := birth_control.get("pill"):
            health_context_lines.append(f"Birth Control - Pill: {bc_pill}")
        if bc_iud := birth_control.get("iud"):
            health_context_lines.append(f"Birth Control - IUD: {bc_iud}")
        if bc_other := birth_control.get("other_methods"):
            health_context_lines.append(
                f"Birth Control - Other Methods: {', '.join(bc_other)}"
            )
        if bc_perm := birth_control.get("permanent"):
            health_context_lines.append(
                f"Birth Control - Permanent: {', '.join(bc_perm)}"
            )
    elif isinstance(birth_control, list) and birth_control:
        # Legacy fallback if an older payload arrives with a flat list
        health_context_lines.append(f"Birth Control: {', '.join(birth_control)}")

    if fertility_treatments := health_profile.get("fertility_treatments"):
        health_context_lines.append(
            f"Fertility Treatments (last 3 months): {', '.join(fertility_treatments)}"
        )

    recent_context = health_profile.get("recent_vaginal_context")
    if isinstance(recent_context, dict):
        for key, label in (
            ("vaginal_products", "Recent Vaginal Products"),
            ("sex_fluids", "Recent Sexual Activity"),
            ("menstrual_blood_or_spotting", "Recent Bleeding/Spotting"),
        ):
            rendered = render_selection(recent_context.get(key))
            if rendered is not None:
                health_context_lines.append(f"{label}: {rendered}")

    notes = health_profile.get("notes")

    # ── Assemble ──
    # No-profile branch: when nothing but pH is present, return the bare pH line
    # (no section headers) so the prompt's NO-PROFILE detection still triggers.
    if not (symptom_lines or baseline_lines or health_context_lines or notes):
        return f"pH Value: {ph_value}"

    blocks = [f"[Your pH]\npH Value: {ph_value}"]
    if symptom_lines:
        blocks.append("[Symptoms]\n" + "\n".join(symptom_lines))
    if baseline_lines:
        blocks.append("[Your personal baseline]\n" + "\n".join(baseline_lines))
    if health_context_lines:
        blocks.append("[Health context]\n" + "\n".join(health_context_lines))
    if notes:
        blocks.append(
            "[Additional notes — unclassified; do NOT auto-treat as a symptom]\n"
            f"{notes}"
        )
    return "\n\n".join(blocks)
