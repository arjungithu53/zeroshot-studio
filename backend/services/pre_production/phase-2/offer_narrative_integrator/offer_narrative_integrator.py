import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Literal

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")

logger = logging.getLogger("zeroshot.phase2.offer_narrative_integrator")


class ConceptPortfolioUpdate(BaseModel):
    concept_id: str = Field(description="Concept identifier.")
    offer_placement: Optional[str] = Field(default=None, description="Offer placement integrated into narrative beat.")
    cta_framing: str = Field(description="Call-to-action framing style for the concept.")


class OfferIntegrationLogEntry(BaseModel):
    concept_id: str = Field(description="Concept identifier.")
    original_placement: Optional[str] = Field(default=None, description="Original offer placement before refinement.")
    refined_placement: Optional[str] = Field(default=None, description="Refined placement after integration.")
    cta_type_reframe: str = Field(description="How CTA type was reframed for narrative fit.")


class FlaggedConceptEntry(BaseModel):
    concept_id: str = Field(description="Concept identifier.")
    issue: str = Field(description="Why offer could not land cleanly inside allotted offer seconds.")
    workaround: str = Field(description="Creative workaround to preserve narrative coherence.")


class OfferNarrativeIntegratorResult(BaseModel):
    status: Literal["completed", "skipped", "error"] = Field(default="completed")
    reason: Optional[str] = Field(default=None, description="Reason when skipped or errored.")
    concept_portfolio_updates: List[ConceptPortfolioUpdate] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Integration reasoning and applied principles.")
    offer_integration_log: List[OfferIntegrationLogEntry] = Field(default_factory=list)
    flagged_concepts: List[FlaggedConceptEntry] = Field(default_factory=list)


def _clean_json_string(json_str: str) -> str:
    cleaned = json_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _get_nested(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    if current is None:
        return default
    return current


def _apply_updates_to_portfolio(
    concept_portfolio: List[Dict[str, Any]],
    updates: List[ConceptPortfolioUpdate],
) -> List[Dict[str, Any]]:
    updates_by_id = {u.concept_id: u for u in updates}
    merged: List[Dict[str, Any]] = []

    for index, concept in enumerate(concept_portfolio):
        concept_item = dict(concept)
        concept_id = str(concept_item.get("concept_id", "")).strip()
        selected_update = updates_by_id.get(concept_id)

        if selected_update is None and updates:
            logger.warning(
                "Agent 21: Missing concept_id match for portfolio index=%s (concept_id=%s). Falling back by index.",
                index,
                concept_id,
            )
            if index < len(updates):
                selected_update = updates[index]

        if selected_update is not None:
            concept_item["offer_placement"] = selected_update.offer_placement
            concept_item["cta_framing"] = selected_update.cta_framing

        merged.append(concept_item)

    return merged


async def run_offer_narrative_integrator_agent(project_id: str, db: Any) -> OfferNarrativeIntegratorResult:
    logger.info("Initializing Agent 21 (Offer Narrative Integrator) for project_id=%s", project_id)
    start_time = time.time()

    try:
        project_obj_id = ObjectId(project_id)
    except Exception as exc:
        logger.error("Agent 21: Invalid project_id format for ObjectId conversion. project_id=%s error=%s", project_id, exc)
        raise ValueError(f"Invalid project_id {project_id}")

    logger.info("Agent 21: Fetching data for project_id=%s", project_id)
    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": project_obj_id})
    if not project_doc:
        logger.error("Agent 21: Project document not found in %s for project_id=%s", PROJECTS_COLLECTION, project_id)
        raise ValueError(f"Project not found in {PROJECTS_COLLECTION}")

    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}

    strategy_agents = strategy_doc.get("agents", {})
    value_prop_and_offer = strategy_agents.get("value_prop_and_offer", {})
    creative_brief = _get_nested(strategy_agents, ["insight_validation", "creative_brief"], {})

    price_and_offer = project_doc.get("price_and_offer")
    offer_hook = value_prop_and_offer.get("offer_hook", "")
    rational_bridge = value_prop_and_offer.get("rational_bridge", "")
    barrier_addressed = value_prop_and_offer.get("barrier_addressed")
    primary_benefit = value_prop_and_offer.get("primary_benefit", "")
    support_points = creative_brief.get("support_points", []) if isinstance(creative_brief, dict) else []

    concept_portfolio = ideation_doc.get("concept_portfolio", [])
    narrative_budget = ideation_doc.get("narrative_budget", {})
    offer_seconds = narrative_budget.get("offer_seconds") if isinstance(narrative_budget, dict) else None
    video_type_final = ideation_doc.get("video_type_final", "")
    selected_archetype = ideation_doc.get("selected_archetype", {})

    logger.info("Agent 21: Extracted key inputs from DB docs.")
    logger.info(
        "Agent 21: Input summary | price_and_offer_present=%s offer_hook_present=%s rational_bridge_present=%s barrier_addressed=%s primary_benefit_present=%s support_points_count=%s concept_count=%s offer_seconds=%s video_type_final=%s",
        not _is_blank(price_and_offer),
        not _is_blank(offer_hook),
        not _is_blank(rational_bridge),
        barrier_addressed,
        not _is_blank(primary_benefit),
        len(support_points) if isinstance(support_points, list) else 0,
        len(concept_portfolio) if isinstance(concept_portfolio, list) else 0,
        offer_seconds,
        video_type_final,
    )

    if not isinstance(concept_portfolio, list) or not concept_portfolio:
        logger.error("Agent 21: concept_portfolio missing or empty in %s for project_id=%s", IDEATION_COLLECTION, project_id)
        raise ValueError("concept_portfolio is missing in ideation document")

    if _is_blank(price_and_offer):
        logger.info("Agent 21: BYPASS triggered because projects.price_and_offer is null/empty for project_id=%s", project_id)

        bypass_updates: List[ConceptPortfolioUpdate] = []
        for index, concept in enumerate(concept_portfolio):
            concept_id = str(concept.get("concept_id") or f"concept_{index + 1}")
            bypass_updates.append(
                ConceptPortfolioUpdate(
                    concept_id=concept_id,
                    offer_placement=None,
                    cta_framing="brand_only",
                )
            )

        merged_portfolio = _apply_updates_to_portfolio(concept_portfolio, bypass_updates)
        result = OfferNarrativeIntegratorResult(
            status="skipped",
            reason="projects.price_and_offer is null; bypass logic applied",
            concept_portfolio_updates=bypass_updates,
            reasoning="No explicit commercial offer was provided. Offer integration was bypassed and CTA framing set to brand_only for all concepts.",
            offer_integration_log=[],
            flagged_concepts=[],
        )

        logger.info("Agent 21: Updating IDEATION and PIPELINE collections for bypass outcome...")
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {
                "$set": {
                    "concept_portfolio": merged_portfolio,
                    "status.offer_narrative_integrator": "skipped",
                    "updated_at": time.time(),
                }
            },
            upsert=True,
        )

        duration = time.time() - start_time
        pipeline_log = {
            "agent_id": 21,
            "agent_name": "offer_narrative_integrator",
            "status": "skipped",
            "timestamp": time.time(),
            "execution_time_sec": round(duration, 2),
            "duration_sec": round(duration, 2),
            "reasoning": result.reasoning,
            "reason": result.reason,
            "output": {
                "concept_portfolio_updates": [u.model_dump() for u in bypass_updates],
                "offer_integration_log": [],
                "flagged_concepts": [],
            },
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )

        logger.info("Agent 21: Successfully updated IDEATION and PIPELINE collections for bypass outcome.")
        logger.info("Agent 21: Execution completed in %.2fs for project_id=%s", duration, project_id)
        return result

    prompt_payload = {
        "price_and_offer": price_and_offer,
        "offer_hook": offer_hook,
        "rational_bridge": rational_bridge,
        "barrier_addressed": barrier_addressed,
        "primary_benefit": primary_benefit,
        "creative_brief_support_points": support_points,
        "concept_portfolio": concept_portfolio,
        "narrative_budget_offer_seconds": offer_seconds,
        "video_type_final": video_type_final,
        "selected_archetype": selected_archetype,
    }

    prompt = f"""
You are Agent 21a: offer_narrative_integrator.

RUN CONDITION:
- Execute only when projects.price_and_offer is not null.
- If no offer is present, bypass (already handled by caller-side control flow).

Purpose:
Convert the commercial offer into a narrative device organic to each concept's archetype.
The offer must not feel like a transactional interruption.
Specify the minimum visual/text treatment needed to land the offer clearly within narrative_budget.offer_seconds.

Prompt Logic:
Review every concept in concept_portfolio and ensure the commercial offer is integrated as a narrative device, not appended as a transaction.

For each concept, evaluate:
1) Whether the offer appears as a native narrative consequence of emotional payoff or as a separate commercial beat.
2) How to reframe offer mechanics as an archetype-consistent narrative device.
3) The minimum visual/text treatment needed for clarity without cognitive overload.
   Include required on-screen text elements, max duration of the offer beat, and whether it should be voiced, captioned, or visual.

Flagging rule:
Flag any concept where the offer cannot be communicated clearly within narrative_budget_offer_seconds without breaking narrative coherence.
For flagged concepts, provide a creative workaround.

barrier_addressed calibration:
Use barrier_addressed from strategy.agents.value_prop_and_offer to calibrate how explicitly the offer must de-risk the purchase decision in each concept.
If barrier_addressed identifies a specific hesitation, include at least one sensory proof or signal adjacent to the offer beat.

Required Output:
Return JSON matching the provided schema with these fields:
- concept_portfolio_updates: list of objects per concept with concept_id, offer_placement, cta_framing
- reasoning: string
- offer_integration_log: list of objects with concept_id, original_placement, refined_placement, cta_type_reframe
- flagged_concepts: list of objects with concept_id, issue, workaround

INPUT JSON:
{json.dumps(prompt_payload, indent=2)}
"""

    invoke_start = time.time()
    logger.info(f"Agent 21: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 21: Gemini Client instantiated. Sending prompt...")

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": OfferNarrativeIntegratorResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )

        api_duration = time.time() - invoke_start
        logger.info(f"Agent 21: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 21: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 21: Successfully parsed JSON response.")

        result = OfferNarrativeIntegratorResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 21: Successfully validated structured output with Pydantic.")
    except Exception as exc:
        logger.error("Agent 21: Error during Gemini inference or JSON parsing for project_id=%s error=%s", project_id, exc)
        raise

    merged_portfolio = _apply_updates_to_portfolio(concept_portfolio, result.concept_portfolio_updates)

    logger.info("Agent 21: Updating IDEATION and PIPELINE collections...")
    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {
                "$set": {
                    "concept_portfolio": merged_portfolio,
                    "status.offer_narrative_integrator": "completed",
                    "updated_at": time.time(),
                }
            },
            upsert=True,
        )

        total_duration = time.time() - start_time
        pipeline_log = {
            "agent_id": 21,
            "agent_name": "offer_narrative_integrator",
            "status": "completed",
            "timestamp": time.time(),
            "execution_time_sec": round(total_duration, 2),
            "duration_sec": round(total_duration, 2),
            "reasoning": result.reasoning,
            "output": {
                "concept_portfolio_updates": [u.model_dump() for u in result.concept_portfolio_updates],
                "offer_integration_log": [entry.model_dump() for entry in result.offer_integration_log],
                "flagged_concepts": [entry.model_dump() for entry in result.flagged_concepts],
            },
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )
        logger.info("Agent 21: Successfully updated IDEATION and PIPELINE collections.")
    except Exception as exc:
        logger.error("Agent 21: Error writing DB updates for project_id=%s error=%s", project_id, exc)
        raise

    total_duration = time.time() - start_time
    logger.info("Agent 21: Execution completed in %.2fs for project_id=%s", total_duration, project_id)
    return result
