"""Pydantic schemas for request/response validation.

Input shape mirrors the 3-step Figma form (About you / Cycle & meds / Symptoms)
plus two optional sub-forms (Birth control + Fertility) that are shown inline
in the Streamlit testing app. The real Figma flow shows them as conditional
fork screens, but for internal testing they're flat.

`extra="ignore"` is set on QueryRequest so unknown fields don't break parsing.
A `model_validator` (`_normalize_fe_payload`) acts as an anti-corruption layer:
it reshapes the live FE payload into our canonical fields (lifts recent vaginal
context out of `symptoms`, maps `birth_control.general` -> `recent_changes`, and
merges `hormone_therapy` / `hrt` into `medications`) before field validation.
Genuinely unused extras (e.g. `fertility_journey.current_status`) are dropped.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BirthControlInfo(BaseModel):
    """Birth control sub-form.

    Shown inline in Streamlit when 'Birth control' is selected in
    QueryRequest.medications. The real Figma flow opens this as a fork screen.
    """

    recent_changes: list[str] = Field(
        default_factory=list,
        description=(
            "Recent birth-control-related changes (multi-select): "
            "'Stopped birth control in the last 3 months', "
            "'Morning after pill / emergency contraception in the last 7 days'"
        ),
    )
    pill: str | None = Field(
        None, description="Pill (single-select): Combined pill, Progestin-only pill"
    )
    iud: str | None = Field(
        None, description="IUD (single-select): Hormonal IUD, Copper IUD"
    )
    other_methods: list[str] = Field(
        default_factory=list,
        description=(
            "Other hormonal methods (multi-select): Contraceptive implant, "
            "Contraceptive injection, Vaginal ring, Patch"
        ),
    )
    permanent: list[str] = Field(
        default_factory=list,
        description="Permanent methods (multi-select): Tubal ligation",
    )

    model_config = ConfigDict(extra="ignore")


class FertilityJourneyInfo(BaseModel):
    """Fertility journey sub-form.

    Shown inline in Streamlit when 'Fertility treatment' is selected in
    QueryRequest.medications. The old `current_status` field has moved to
    QueryRequest.life_stage and is no longer captured here.
    """

    fertility_treatments: list[str] = Field(
        default_factory=list,
        description=(
            "Treatments in the last 3 months (multi-select): Ovulation induction, "
            "Intrauterine insemination (IUI), In vitro fertilisation (IVF), "
            "Egg freezing stimulation, Luteal progesterone"
        ),
    )

    model_config = ConfigDict(extra="ignore")


class SymptomsInfo(BaseModel):
    """Current symptoms, split into 4 semantic groups + free-text notes.

    The split is preserved through to the prompt so the model knows which
    symptom is which type — flattening loses that signal.
    """

    discharge: list[str] = Field(
        default_factory=list,
        description=(
            "Discharge (multi-select): No discharge, Creamy, Sticky, Egg white, "
            "Clumpy white, Grey and watery, Yellow / Green, Red / Brown"
        ),
    )
    vulva_vagina: list[str] = Field(
        default_factory=list,
        description=(
            "Vulva & Vagina (multi-select): Dry, Itchy"
        ),
    )
    smell: list[str] = Field(
        default_factory=list,
        description=(
            'Smell (multi-select): None, Strong and unpleasant ("fishy"), '
            "Sour, Chemical-like, Very strong or rotten"
        ),
    )
    urine: list[str] = Field(
        default_factory=list,
        description=(
            "Urine (multi-select): More frequent than usual, Burning sensation"
        ),
    )
    notes: str | None = Field(
        None,
        description=(
            "Free-text notes about additional symptoms or how the user has been feeling"
        ),
    )

    model_config = ConfigDict(extra="ignore")


class RecentVaginalContextInfo(BaseModel):
    """Recent vaginal context (collected on the last-step screen).

    Products, sex/fluid exposure, and recent bleeding/spotting. Per the v2 spec
    this maps to the 'Health context' result section, not Symptoms.
    """

    vaginal_products: list[str] = Field(
        default_factory=list,
        description=(
            "Vaginal products (multi-select): Intimate wash, Scented soap near "
            "the vulva, Douche, Vaginal deodorant, Lubricant, Spermicide, Boric "
            "acid, Vaginal probiotic, Vaginal moisturizer, "
            "Vaginal medication / suppository"
        ),
    )
    sex_fluids: list[str] = Field(
        default_factory=list,
        description=(
            "Sex & fluids (multi-select): Sex without a condom / barrier, Sex "
            "with a condom / barrier, Semen may have been present"
        ),
    )
    menstrual_blood_or_spotting: list[str] = Field(
        default_factory=list,
        description=(
            "Menstrual blood or spotting (multi-select): Period started, Period "
            "just ended, Spotting, Blood may have been present"
        ),
    )

    model_config = ConfigDict(extra="ignore")


class QueryRequest(BaseModel):
    """Request schema for pH analysis query.

    Shape mirrors the 3-step Figma input form plus two optional sub-forms.
    All non-pH fields are optional — backend tolerates a bare-minimum
    pH-only submission (it produces the 'no profile' branch output).
    """

    # ── Required ──
    ph_value: float = Field(
        ..., ge=0.0, le=14.0, description="pH value from test strip (0-14) - REQUIRED"
    )

    # ── Optional: Conversation context (not part of the Figma form) ──
    user_message: str | None = Field(
        None,
        description=(
            "User's actual question (for follow-up turns). If None, the API "
            "auto-generates the initial query from the health profile."
        ),
    )
    session_id: str | None = Field(
        None,
        description="Session ID for conversation continuity. If None, creates new session.",
    )

    # ── Step 1: About you ──
    age: int | None = Field(None, ge=0, le=120, description="User's age")
    life_stage: list[str] = Field(
        default_factory=list,
        description=(
            "Life stage (multi-select): None, Pregnant, Trying to conceive, "
            "Perimenopause, Menopause, Postmenopause. Accepts None, a single "
            "string, or a list of strings; normalized to list[str] by the "
            "validator below."
        ),
    )
    ethnic_backgrounds: list[str] = Field(
        default_factory=list,
        description=(
            "Ethnic backgrounds (multi-select): African / Black, North African, "
            "Arab, Middle Eastern, East Asian, South Asian, Southeast Asian, "
            "Central Asian / Caucasus, Latin American / Latina / Latinx / Hispanic, "
            "Sinti / Roma, White / Caucasian / European, Mixed / Multiple ancestry, Other"
        ),
    )
    ethnic_background_other: str | None = Field(
        None,
        description=(
            "Free-text capture when 'Other' is selected in ethnic_backgrounds "
            "(e.g. 'Afro-Caribbean')"
        ),
    )

    # ── Step 2: Cycle, hormones & medications ──
    menstrual_cycle: list[str] = Field(
        default_factory=list,
        description=(
            "Menstrual cycle (multi-select): Regular, Irregular, "
            "No period for 12+ months, Currently pregnant, Postpartum, "
            "Never had a period, Polycystic ovary syndrome (PCOS). Accepts None, "
            "a single string, or a list of strings; normalized to list[str] by "
            "the validator below."
        ),
    )
    medications: list[str] = Field(
        default_factory=list,
        description=(
            "Medications or hormone treatment (multi-select): None, Antibiotics, "
            "Antifungals, Birth control, Fertility treatment, Estrogen only, "
            "Estrogen + progestin, Estrogen blocker, Puberty blocker, Testosterone"
        ),
    )
    # NOTE: "Amenorhea" below matches the FE label verbatim (sic — upstream
    # misspelling of "Amenorrhea"). Kept as-is to align 1:1 with the frontend;
    # do not "correct" it unless the FE label is fixed first.
    diagnoses: list[str] = Field(
        default_factory=list,
        description=(
            "Diagnoses related to hormones (multi-select): None, Adenomyosis, "
            "Amenorhea, Cushing's syndrome, Diabetes, Endometriosis, Intersex, "
            "Thyroid disorder, Uterine fibroids, Premature ovarian insufficiency (POI)"
        ),
    )

    # ── Optional sub-form: Birth control details ──
    # (Shown inline in Streamlit when 'Birth control' is in medications)
    birth_control: BirthControlInfo | None = Field(
        None, description="Birth control sub-form details"
    )

    # ── Optional sub-form: Fertility journey ──
    # (Shown inline in Streamlit when 'Fertility treatment' is in medications)
    fertility_journey: FertilityJourneyInfo | None = Field(
        None, description="Fertility journey sub-form details"
    )

    # ── Step 3: Symptoms ──
    symptoms: SymptomsInfo | None = Field(None, description="Current symptoms")

    # ── Last step: Recent vaginal context ──
    recent_vaginal_context: RecentVaginalContextInfo | None = Field(
        None,
        description=(
            "Recent vaginal context sub-form (vaginal products, sex/fluids, "
            "bleeding/spotting). Maps to the Health context result section."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_fe_payload(cls, data: Any) -> Any:
        """Anti-corruption layer: reshape the live FE payload into our canonical
        fields before field validation.

        The FE form diverges from our schema in a few confirmed places (verified
        against captured payloads, 2026-06-08). Translating here keeps the rest of
        the pipeline (route, build_health_context, prompt) on our own vocabulary:

        - Recent vaginal context is nested INSIDE `symptoms`, and spotting is its
          own key. Lift `symptoms.vaginal_products` / `sex_fluids` / `spotting`
          out into a top-level `recent_vaginal_context`
          (spotting -> menstrual_blood_or_spotting). Only fills what the caller
          did not already send at the top level.
        - Birth-control "recent changes" arrives as `birth_control.general`
          (a single string). Map it to `recent_changes` (a list).
        - The medications screen posts into `hormone_therapy` / `hrt` (legacy
          keys), not `medications`. Merge all three into `medications`.

        Unknown extras (e.g. `fertility_journey.current_status`) are left for
        extra="ignore" to drop.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)  # copy: never mutate the caller's payload

        # 1) Lift recent vaginal context out of `symptoms` into its own object.
        symptoms = data.get("symptoms")
        if isinstance(symptoms, dict):
            rvc = dict(data.get("recent_vaginal_context") or {})
            for fe_key, canonical_key in (
                ("vaginal_products", "vaginal_products"),
                ("sex_fluids", "sex_fluids"),
                ("spotting", "menstrual_blood_or_spotting"),
            ):
                if symptoms.get(fe_key) and not rvc.get(canonical_key):
                    rvc[canonical_key] = symptoms[fe_key]
            if rvc:
                data["recent_vaginal_context"] = rvc

        # 2) Birth control: FE `general` (single string) -> `recent_changes` (list).
        birth_control = data.get("birth_control")
        if (
            isinstance(birth_control, dict)
            and birth_control.get("general")
            and not birth_control.get("recent_changes")
        ):
            birth_control = dict(birth_control)
            birth_control["recent_changes"] = [birth_control["general"]]
            data["birth_control"] = birth_control

        # 3) Medications: FE posts into `hormone_therapy` / `hrt`, not
        #    `medications`. Merge all three (de-duplicated, order preserved).
        merged_medications: list[Any] = []
        for source_key in ("medications", "hormone_therapy", "hrt"):
            values = data.get(source_key)
            if isinstance(values, list):
                for value in values:
                    if value not in merged_medications:
                        merged_medications.append(value)
        if merged_medications:
            data["medications"] = merged_medications

        return data

    @field_validator("life_stage", mode="before")
    @classmethod
    def _coerce_life_stage_to_list(cls, value: Any) -> Any:
        """Normalize None | str | list[str] to list[str].

        Backward-compat shim: live FE posts a list, legacy Streamlit posts a
        single string, and explicit None / omission should be treated as "no
        selection" (empty list). The route layer's SENTINEL filter handles
        dropping any "None" entries afterward.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        return value

    @field_validator("menstrual_cycle", mode="before")
    @classmethod
    def _coerce_menstrual_cycle_to_list(cls, value: Any) -> Any:
        """Normalize None | str | list[str] to list[str].

        Same coercion rules as `_coerce_life_stage_to_list`. See that
        validator for the rationale.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        return value

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "ph_value": 4.8,
                    "age": 28,
                    "life_stage": "Trying to conceive",
                    "ethnic_backgrounds": ["South Asian"],
                    "menstrual_cycle": "Irregular",
                    "medications": ["Birth control"],
                    "diagnoses": ["Endometriosis"],
                    "birth_control": {"pill": "Combined pill"},
                    "symptoms": {
                        "discharge": ["Creamy"],
                        "vulva_vagina": ["Itchy"],
                        "smell": [],
                        "urine": [],
                        "notes": "Feeling stressed lately",
                    },
                }
            ]
        },
    )


class CitationResponse(BaseModel):
    """
    DECIDE-03 wire shape: full citation card with server-rendered URL fields.
    Bare identifiers (pmid, pmcid, doi) are NOT on the wire — clients use the
    rendered URL fields directly as hrefs.
    """

    id: int = Field(
        ...,
        description=(
            "Sequential citation id (1..N). Maps to [N] markers in the deep-dive section "
            "fields and overview. Assigned by the backend renumbering pass in routes/query.py."
        ),
    )
    title: str = Field(
        ...,
        description="Paper title from papers.title (NOT NULL per DECIDE-01 E-REQ-2).",
    )
    nlm_reference: str | None = Field(
        None,
        description=(
            "NLM-style citation string (authors, journal, date, volume, pages — "
            "WITHOUT title; sourced from papers.nlm_reference). None when NULL in DB."
        ),
    )
    pmid_url: str | None = Field(
        None,
        description=(
            "Full PubMed URL rendered server-side from "
            "https://pubmed.ncbi.nlm.nih.gov/{pmid}/. None when papers.pmid is NULL."
        ),
    )
    pmcid_url: str | None = Field(
        None,
        description=(
            "Full PMC URL rendered server-side from "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/. None when papers.pmcid is NULL."
        ),
    )
    doi_url: str | None = Field(
        None,
        description=(
            "Full DOI URL rendered server-side from "
            "https://doi.org/{doi}. None when papers.doi is NULL."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "title": "Vaginal pH and the Lactobacillus-dominant microbiome",
                "nlm_reference": "Lin Y, Chen H, et al. J Womens Health. 2021;30(4):512-520.",
                "pmid_url": "https://pubmed.ncbi.nlm.nih.gov/33712345/",
                "pmcid_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8123456/",
                "doi_url": "https://doi.org/10.1089/jwh.2020.0123",
            }
        }
    )


class QueryResponse(BaseModel):
    """Response schema for pH analysis query.

    The 5 deep-dive section fields (your_ph, your_symptoms, your_personal_baseline,
    your_health_context, next_steps) each carry one section's markdown content
    WITHOUT a `## header` (the field name is the section). Optional sections are
    `None` when not produced (e.g. bare pH-only requests).

    `overview` carries the bullet-list summary. All section fields plus overview
    share citation numbering ([N] markers).
    """

    session_id: str = Field(..., description="Unique session identifier")
    ph_value: float = Field(..., description="The analyzed pH value")
    your_ph: str = Field(
        ...,
        description=(
            "'Your pH' section content (markdown, no ## header). REQUIRED — always "
            "populated. Contains [N] citation markers sharing numbering with overview "
            "and the other section fields. Renumbered server-side."
        ),
    )
    your_symptoms: str | None = Field(
        default=None,
        description=(
            "'Your symptoms' section content (markdown, no ## header). None when the "
            "patient reported no symptoms or sent a bare pH-only request. Contains [N] "
            "markers when present."
        ),
    )
    your_personal_baseline: str | None = Field(
        default=None,
        description=(
            "'Your personal baseline' section content (markdown, no ## header). None "
            "when no profile was provided (bare pH-only request). Contains [N] markers "
            "when present."
        ),
    )
    your_health_context: str | None = Field(
        default=None,
        description=(
            "'Your health context' section content (markdown, no ## header). None when "
            "no profile was provided. Contains [N] markers when present."
        ),
    )
    next_steps: str = Field(
        ...,
        description=(
            "'Next steps' section content (markdown, no ## header). REQUIRED — always "
            "populated. Only field where doctor / consultation / referral language is "
            "allowed. Contains [N] markers."
        ),
    )
    overview: str = Field(
        ...,
        description=(
            "Markdown bullet list summarizing the deep-dive sections. Contains [N] "
            "citation markers sharing numbering with the 5 section fields. Populated "
            "by MedicalResponse.overview after the backend renumbering pass."
        ),
    )
    disclaimers: str = Field(..., description="Medical disclaimers")
    citations: list[CitationResponse] = Field(
        default_factory=list, description="Research paper citations"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "session_id": "abc123",
                    "ph_value": 4.8,
                    "your_ph": "Your pH of 4.8 is slightly above the typical healthy range [1]...",
                    "your_symptoms": "Based on the creamy discharge you reported... **Watch for:** new fishy odor or itching.",
                    "your_personal_baseline": "At 28 and trying to conceive, your cycle phase can shift pH...",
                    "your_health_context": "With endometriosis and combined pill use, vaginal pH may trend...",
                    "next_steps": "Retest in 7 days. If your next test is also elevated, consider consulting a healthcare provider.",
                    "overview": "- **Your pH is slightly above the usual range** — 4.8 is borderline [1].\n- **Symptoms:** ...",
                    "disclaimers": "This is not medical advice...",
                    "citations": [
                        {
                            "id": 1,
                            "title": "Vaginal pH and the Lactobacillus-dominant microbiome",
                            "nlm_reference": "Lin Y, Chen H, et al. J Womens Health. 2021;30(4):512-520.",
                            "pmid_url": "https://pubmed.ncbi.nlm.nih.gov/33712345/",
                            "pmcid_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8123456/",
                            "doi_url": "https://doi.org/10.1089/jwh.2020.0123",
                        }
                    ],
                    "processing_time_ms": 1500,
                }
            ]
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
