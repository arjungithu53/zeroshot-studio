import json
import logging
import os
import time
from typing import List

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

logger = logging.getLogger("zeroshot.phase3.offer_integration_planner")


def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")


class OfferConstraints(BaseModel):
    vo_prohibition: str = Field(description="Hard prohibition for voice-over during offer window.")
    vo_language_directive: str = Field(description="Required language register and style for voice-over.")
    visual_directive: str = Field(description="Binding visual treatment directive for offer communication.")
    cta_channel_rules: List[str] = Field(description="Per-channel CTA rules that downstream agents must obey.")
    text_super_max_words: int = Field(description="Maximum allowed word count for text super in offer window.")
    risk_mitigations: List[str] = Field(description="One mitigation directive per risk tag.")


class OfferIntegrationPlannerResult(BaseModel):
    offer_constraints: OfferConstraints
    reasoning: str
    status: str = "pending"


async def run_offer_integration_planner_agent(project_id: str, db) -> OfferIntegrationPlannerResult:
    agent_key = "offer_integration_planner"
    run_start = time.time()
    logger.info(f"Initializing Agent [2]... | project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error("Agent [2]: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"[{agent_key}] Fetching data for project_id={project_id}")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")

        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        if not strategy_doc:
            raise ValueError(f"Strategy document for '{project_id}' not found")

        pipeline_doc = await db[PIPELINE_COLLECTION].find_one({"project_id": str(project_id)})
        if not pipeline_doc:
            logger.info(f"[{agent_key}] Pipeline document not found. Continuing with empty pipeline context.")
            pipeline_doc = {}

        script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
        if not script_doc:
            logger.info(f"[{agent_key}] Script document not found. Continuing with empty script context.")
            script_doc = {}

        phase_2_output = ideation_doc.get("phase_2_output", ideation_doc)
        approved_concepts = phase_2_output.get("approved_concepts", ideation_doc.get("approved_concepts", []))
        if not approved_concepts:
            raise ValueError("No approved concepts found in ideation document")

        concept_index = 0
        concept = approved_concepts[concept_index]
        concept_id = str(concept.get("concept_id") or f"concept_{concept_index + 1}")
        concept_offer_placement = concept.get("offer_placement")
        concept_cta_framing = concept.get("cta_framing")
        concept_risk_tags = concept.get("risk_tags", [])

        # Format-group-aware archetype/motif extraction
        format_group = phase_2_output.get("format_group", ideation_doc.get("format_group", "N"))
        is_visual = format_group == "V"
        if is_visual:
            selected_motif = phase_2_output.get("selected_visual_motif", ideation_doc.get("selected_visual_motif", {}))
            concept_archetype = selected_motif.get("selected_motif", concept.get("visual_motif", ""))
            micro_policy = selected_motif.get("visual_micro_policy", "")
            failure_modes = selected_motif.get("failure_modes", [])
            archetype_label = "VISUAL MOTIF GOVERNING RULES"
        else:
            selected_archetype = phase_2_output.get("selected_archetype", ideation_doc.get("selected_archetype", {}))
            concept_archetype = concept.get("archetype", "")
            micro_policy = selected_archetype.get("micro_policy", "")
            failure_modes = selected_archetype.get("failure_modes", [])
            archetype_label = "ARCHETYPE GOVERNING RULES"

        price_and_offer = project_doc.get("price_and_offer")
        product_url = project_doc.get("product_url") or "Not provided"

        master_timeline = script_doc.get("master_timeline", {})
        master_timeline_windows = master_timeline.get("windows", []) if isinstance(master_timeline, dict) else []

        if not concept_offer_placement:
            raise ValueError("concept.offer_placement is missing for selected approved concept")
        if not concept_cta_framing:
            raise ValueError("concept.cta_framing is missing for selected approved concept")
        if not price_and_offer:
            raise ValueError("projects.price_and_offer is missing in projects document")
        if not master_timeline_windows:
            raise ValueError("master_timeline.windows is missing in script document")

        logger.info(
            f"[{agent_key}] Extracted key inputs: concept_id={concept_id}, format_group={format_group}, "
            f"offer_placement_present={bool(concept_offer_placement)}, cta_framing_present={bool(concept_cta_framing)}, "
            f"archetype/motif='{concept_archetype}', risk_tags_count={len(concept_risk_tags) if isinstance(concept_risk_tags, list) else 0}, "
            f"micro_policy_present={bool(micro_policy)}, "
            f"failure_modes_count={len(failure_modes) if isinstance(failure_modes, list) else 0}, "
            f"price_and_offer_present={bool(price_and_offer)}, master_timeline_windows_count={len(master_timeline_windows)}, "
            f"pipeline_agent_logs_count={len(pipeline_doc.get('agent_logs', []))}"
        )

        prompt = f"""
You are a pre-scriptwriting offer integration strategist for short-form branded video. Your job runs before any creative writing happens. You identify and resolve the fundamental tension between a concept's emotional archetype and its commercial CTA, then produce a set of binding channel-level directives that every downstream writing agent must follow without deviation.

You are grounded strictly in the inputs provided. You do not write script lines, VO, or visual descriptions. You produce constraints. Think of yourself as the contract that the rest of the pipeline must honour.


CONCEPT OFFER DETAILS:
Offer placement description: {concept_offer_placement}
CTA framing register: {concept_cta_framing}
{"Archetype" if not is_visual else "Visual motif"}: {concept_archetype}

{archetype_label}:
{micro_policy}
Failure modes to avoid: {json.dumps(failure_modes, ensure_ascii=True)}

COMMERCIAL OFFER:
{price_and_offer}

PRODUCT WEBPAGE URL:
{product_url}
(If a product URL is provided, use it to understand the specific product's features and offer language in depth.)

RISK TAGS FROM PHASE 2 (pre-emptive issues identified):
{json.dumps(concept_risk_tags, ensure_ascii=True)}

MASTER TIMELINE (offer window reference):
{json.dumps(master_timeline_windows, ensure_ascii=True)}

TASK:
Step 1 — Identify every channel (VO, visual, text super, audio) where the offer will appear during the offer window. For each channel, determine whether the offer as currently described is tonally consistent with the archetype's micro_policy. Flag any channel where it is not.

Step 2 — For each flagged channel, write a specific binding directive that resolves the conflict. Example: if the archetype demands no transactional urgency but the offer uses "Buy Now", mandate that the word "Buy" is restricted to text super only and VO must use sharing/invitation language.

Step 3 — For each risk_tag, issue one specific mitigation constraint that downstream agents must enforce. Be precise — "avoid clutter" is not a constraint. "Text super maximum 8 words, product visual carries primary offer communication through quantity not text" is a constraint.

Step 4 — Specify the maximum word count for any text super in the offer window.

CONSTRAINTS (apply last):
- Directives must be specific enough that a downstream agent cannot misinterpret them.
- Do not write script content. Do not suggest VO lines. Produce directives only.
- Every directive must be traceable to either the archetype micro_policy or a risk_tag.
- Output as a structured JSON object with fields: vo_prohibition, vo_language_directive, visual_directive, cta_channel_rules (array), text_super_max_words (integer), risk_mitigations (array).

Return only valid JSON matching the schema exactly.
Do not include markdown, prose preamble, or code fences.
Do not add extra fields or omit required fields.
"""

        invoke_start = time.time()
        logger.info(f"Agent [2]: Preparing to call Gemini model={GEMINI_MODEL}...")
        try:
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
            )
            logger.info(f"Agent [2]: Gemini Client instantiated. Sending prompt...")

            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": OfferIntegrationPlannerResult.model_json_schema(),
                    "automatic_function_calling": {"disable": True},
                }
            )

            api_duration = time.time() - invoke_start
            logger.info(f"Agent [2]: Gemini API call completed in {api_duration:.2f}s")
            logger.debug(f"Agent [2]: Raw response length={len(response.text)} chars")

            cleaned_json = _clean_json_string(response.text)
            parsed_data = json.loads(cleaned_json)
            logger.info(f"Agent [2]: Successfully parsed JSON response.")

            result = OfferIntegrationPlannerResult(**parsed_data)
            result.status = "completed"
            logger.info(f"Agent [2]: Successfully validated structured output with Pydantic.")
        except Exception as llm_error:
            if "503" in str(llm_error) or "429" in str(llm_error):
                logger.error(f"Gemini API overloaded: {llm_error}")
            else:
                logger.error(f"Agent [2]: Gemini invocation failed: {llm_error}", exc_info=True)
            raise

        logger.info("Agent [2]: Updating SCRIPT and PIPELINE collections...")
        await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {
                "$set": {
                    "project_id": str(project_id),
                    "offer_constraints": result.offer_constraints.model_dump(),
                }
            },
            upsert=True,
        )

        pipeline_log = {
            "agent_id": 2,
            "agent_name": "offer_integration_planner",
            "execution_time_utc": time.time(),
            "api_duration_s": round(api_duration, 2),
            "duration_s": round(time.time() - run_start, 2),
            "reasoning": result.reasoning,
            "offer_constraints": result.offer_constraints.model_dump(),
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )

        total_duration = time.time() - run_start
        logger.info("Agent [2]: Successfully updated SCRIPT and PIPELINE collections.")
        logger.info(f"Agent [2]: Complete! Total duration {total_duration:.2f}s")
        return result

    except Exception as e:
        logger.error(f"[{agent_key}] Failed with error: {str(e)}", exc_info=True)
        raise
