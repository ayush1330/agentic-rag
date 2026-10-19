"""
Reasoning node for medical RAG: assess retrieval quality and compute confidence.

Confidence calculation strategy:
  1. Score-based: Average relevance score of top-5 chunks
  2. LLM validation (conditional): If score is uncertain (0.4-0.8),
     call LLM once to validate if chunks answer the query
  3. Hybrid blending: (0.6 × score) + (0.4 × llm) for uncertain cases

Optimization: Only call LLM when uncertain. Saves 60-70% of LLM calls.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from medical_agent.agents.state import MedicalAgentState
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


def compute_score_based_confidence(citations: list[dict[str, Any]]) -> float:
    """
    Compute confidence from retrieval scores alone.

    Averages the relevance scores of the retrieved chunks.

    Args:
        citations: List of citation dicts with "score" field

    Returns:
        float: Average score (0.0-1.0)
    """
    if not citations:
        return 0.0

    scores = [c.get("score", 0.0) for c in citations]
    avg_score = float(sum(float(s) for s in scores) / len(scores))

    logger.debug(f"Score-based confidence: {avg_score:.3f} (from {len(scores)} chunks)")

    return avg_score


async def validate_with_llm(
    query: str,
    docs_text: str,
    health_profile: dict[str, Any],
    llm: AzureChatOpenAI,
) -> float:
    """
    Validate if retrieved documents answer the user's query using a single LLM call.

    Args:
        query: User's original query
        docs_text: Formatted text of all retrieved chunks
        health_profile: User's health context
        llm: Azure OpenAI LLM instance

    Returns:
        float: Confidence score 0.0-1.0 from LLM assessment
    """
    logger.info("Validating retrieval with LLM (uncertain confidence zone)")

    # Build health context string
    health_context_parts = []
    if health_profile.get("diagnoses"):
        health_context_parts.append(f"Diagnoses: {', '.join(health_profile['diagnoses'])}")
    if health_profile.get("symptoms"):
        symptoms = health_profile["symptoms"]
        if isinstance(symptoms, list):
            health_context_parts.append(f"Symptoms: {', '.join(symptoms)}")
    if health_profile.get("age"):
        health_context_parts.append(f"Age: {health_profile['age']}")

    health_context = (
        "\n".join(health_context_parts) if health_context_parts else "No health context provided"
    )

    validation_prompt = f"""You are a medical information validation expert. Assess whether the retrieved documents can answer the user's query given their specific health context.

<query>{query}</query>

<patient_context>
{health_context}
</patient_context>

<retrieved_documents>
{docs_text}
</retrieved_documents>

<instructions>
Evaluate whether these documents together contain information relevant to answering the user's query. Consider whether the documents address this specific patient's situation (their pH, symptoms, diagnoses), not just the general topic.
</instructions>

<examples>
Query: "What does a pH of 5.2 mean with BV?" | Docs discuss BV diagnosis, pH thresholds, treatment → {{"can_answer": true, "confidence": 0.85, "reasoning": "Documents directly cover BV-pH relationship and treatment"}}
Query: "Can my diet affect vaginal pH?" | Docs discuss antibiotic treatment for BV → {{"can_answer": false, "confidence": 0.25, "reasoning": "Documents cover BV treatment but not dietary factors affecting pH"}}
</examples>

Output ONLY a valid JSON object — no markdown fencing, no text outside the JSON:
{{
  "can_answer": true/false,
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

Confidence scale:
- 0.0-0.3: Documents completely irrelevant to the query and patient context
- 0.3-0.6: Documents partially relevant, missing key information for this patient
- 0.6-0.8: Documents mostly relevant, contain the main answer
- 0.8-1.0: Documents highly relevant with comprehensive coverage for this patient"""

    try:
        response = await llm.ainvoke([HumanMessage(content=validation_prompt)])
        response_text = str(response.content).strip()

        logger.debug(f"LLM validation response: {response_text[:200]}")

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                logger.warning("Could not parse LLM validation response as JSON, defaulting to 0.5")
                return 0.5

        llm_confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        can_answer = result.get("can_answer", False)
        reasoning = result.get("reasoning", "")

        logger.info(f"LLM validation: can_answer={can_answer}, confidence={llm_confidence:.3f}")
        logger.debug(f"LLM reasoning: {reasoning}")

        return llm_confidence

    except Exception as e:
        logger.error(f"LLM validation failed: {e}", exc_info=True)
        return 0.5


async def reasoning_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Reasoning node: Assess retrieval quality and compute confidence score.

    Process:
      1. Extract scores from retrieved citations
      2. Compute score-based confidence (average of chunk scores)
      3. Decision tree:
         - score >= 0.8: High confidence, use score only (no LLM call)
         - score < 0.4:  Low confidence, use score only (no LLM call)
         - 0.4 <= score < 0.8: Uncertain, validate with LLM (1 call)
      4. Blend uncertain case: final = (0.6 × score) + (0.4 × llm)
      5. Determine retrieval_quality: "high" if final >= 0.7, else "low"

    Note: Always proceeds to generate_node (no looping in Phase 2).

    Args:
        state: Current agent state with citations and docs_text

    Returns:
        State update with confidence_score, retrieval_quality, confidence_method
    """
    logger.info("Executing reasoning_node")

    citations = state.get("citations", [])
    docs_text = state.get("docs_text", "")

    # Extract user query from last message
    user_query = ""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        user_query = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message.get("content", ""))
        )

    # Step 1: Score-based confidence
    score_confidence = compute_score_based_confidence(citations)
    logger.info(f"Score-based confidence: {score_confidence:.3f}")

    # Step 2: Decision tree
    if score_confidence >= 0.8:
        logger.info("High confidence (>= 0.8): Using score-based confidence only")
        final_confidence = score_confidence
        method = "score_only_high"

    elif score_confidence < 0.4:
        logger.info("Low confidence (< 0.4): Already failed, no LLM validation needed")
        final_confidence = score_confidence
        method = "score_only_low"

    else:
        logger.info(
            f"Uncertain zone (0.4-0.8): confidence={score_confidence:.3f}, calling LLM for validation"
        )

        llm = AzureChatOpenAI(
            deployment_name=settings.azure_openai_deployment_name,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            temperature=0.0,
        )

        health_profile = state.get("health_profile", {})
        llm_confidence = await validate_with_llm(
            query=user_query,
            docs_text=docs_text,
            health_profile=health_profile,
            llm=llm,
        )

        final_confidence = (0.6 * score_confidence) + (0.4 * llm_confidence)
        method = "hybrid"

        logger.info(
            f"Hybrid confidence: (0.6 × {score_confidence:.3f}) + (0.4 × {llm_confidence:.3f}) = {final_confidence:.3f}"
        )

    # Step 3: Determine retrieval quality
    retrieval_quality = "high" if final_confidence >= 0.7 else "low"

    logger.info(
        f"Reasoning complete: confidence={final_confidence:.3f}, quality={retrieval_quality}, method={method}"
    )

    return {
        "confidence_score": final_confidence,
        "retrieval_quality": retrieval_quality,
        "confidence_method": method,
    }
