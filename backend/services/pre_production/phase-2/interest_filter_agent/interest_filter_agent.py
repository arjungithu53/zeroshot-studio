import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional

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

logger = logging.getLogger("zeroshot.phase2.interest_filter_agent")


class ConceptInterestScore(BaseModel):
    concept_id: str = Field(description="Concept identifier.")
    scroll_stop: bool = Field(description="Whether concept is genuinely interesting enough to stop scroll.")
    felt_emotion: bool = Field(description="Whether concept creates a felt emotional response.")
    tell_a_friend: bool = Field(description="Whether concept has a culturally shareable moment.")
    cultural_fit: bool = Field(description="Whether concept feels native to audience cultural world.")
    overall_flag: Literal["boring", "standard", "thumb_stopper"] = Field(
        description="Overall concept flag based on community-first scoring."
    )

class InterestFilterAgentResult(BaseModel):
    status: Literal["completed", "skipped", "error"] = Field(default="completed")
    reason: Optional[str] = Field(default=None, description="Reason when skipped or errored.")
    reasoning: str = Field(default="", description="Overall reasoning for interest filtering decisions.")
    concept_interest_scores: List[ConceptInterestScore] = Field(default_factory=list)
    boring_flags: List[str] = Field(default_factory=list)
    thumb_stopper_flags: List[str] = Field(default_factory=list)


def _clean_json_string(json_str: str) -> str:
    cleaned = json_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


async def run_interest_filter_agent(project_id: str, db: Any) -> InterestFilterAgentResult:
    logger.info("Initializing Agent [24]... | project_id=%s", project_id)
    start_time = time.time()

    try:
        project_obj_id = ObjectId(project_id)
    except Exception as exc:
        logger.error("Agent [24]: Invalid project_id format for ObjectId conversion. project_id=%s error=%s", project_id, exc)
        raise ValueError(f"Invalid project_id {project_id}")

    logger.info("Agent [24]: Fetching data for project_id=%s", project_id)
    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": project_obj_id})
    if not project_doc:
        logger.error("Agent [24]: Project document not found in %s for project_id=%s", PROJECTS_COLLECTION, project_id)
        raise ValueError(f"Project not found in {PROJECTS_COLLECTION}")

    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    _ = await db[PIPELINE_COLLECTION].find_one({"project_id": str(project_id)}) or {}

    concept_portfolio = ideation_doc.get("concept_portfolio", [])
    platform_rules = _safe_dict(ideation_doc.get("platform_rules", {}))

    strategy_agents = _safe_dict(strategy_doc.get("agents", {}))
    strategy_models_root = _safe_dict(strategy_doc.get("strategy_models", {}))
    strategy_models_from_agents = _safe_dict(strategy_agents.get("strategy_models", {}))
    strategy_models = strategy_models_from_agents if strategy_models_from_agents else strategy_models_root

    persona = _safe_dict(strategy_agents.get("audience_persona", {}))
    creative_brief = _safe_dict(strategy_models.get("creative_brief", {}))
    target_audience = creative_brief.get("target_audience", "")

    venn_model = _safe_dict(strategy_models.get("venn_model", {}))
    audience_cares_about = venn_model.get("audience_cares_about", [])
    if not isinstance(audience_cares_about, list):
        audience_cares_about = []

    logger.info("Agent [24]: Extracted key inputs from DB docs.")
    logger.info(
        "Agent [24]: Input summary | concept_count=%s platform_rules_present=%s persona_present=%s target_audience_present=%s audience_cares_about_count=%s",
        len(concept_portfolio) if isinstance(concept_portfolio, list) else 0,
        bool(platform_rules),
        bool(persona),
        bool(str(target_audience).strip()),
        len(audience_cares_about),
    )
    logger.info(
        "Agent [24]: Extracted fields | ideation.concept_portfolio=%s ideation.platform_rules=%s strategy.agents.audience_persona=%s strategy_models.creative_brief.target_audience=%s strategy_models.venn_model.audience_cares_about=%s",
        "present" if isinstance(concept_portfolio, list) and bool(concept_portfolio) else "missing_or_empty",
        "present" if bool(platform_rules) else "missing_or_empty",
        "present" if bool(persona) else "missing_or_empty",
        "present" if bool(str(target_audience).strip()) else "missing_or_empty",
        "present" if bool(audience_cares_about) else "missing_or_empty",
    )

    if not isinstance(concept_portfolio, list) or not concept_portfolio:
        logger.error("Agent [24]: concept_portfolio missing or empty in %s for project_id=%s", IDEATION_COLLECTION, project_id)
        raise ValueError("concept_portfolio is missing in ideation document")

    prompt_payload = {
        "concept_portfolio": concept_portfolio,
        "platform_rules": platform_rules,
        "persona": persona,
        "creative_brief.target_audience": target_audience,
        "audience_cares_about": audience_cares_about,
    }

    prompt = f"""
You are Agent 24: interest_filter_agent.

RUN CONDITION: ALWAYS

Source: Ch.12 'Prime Directive' p.311

Purpose - Grounded in Prime Directive Framework:
Evaluate each concept against three community-first questions:
(1) Would the persona stop scrolling - not because it is loud but because it is genuinely interesting?
(2) Would the persona feel something while watching - pride, recognition, laughter, longing, defiance?
(3) Is there any moment in this concept that the persona would want to describe to a friend - not because the product is good but because the concept itself is culturally interesting?

Flagging logic:
- Concepts that fail all three questions are flagged as "boring".
- Concepts that pass at least two are "standard".
- Concepts that pass all three are flagged as "thumb_stopper".

Community fit check:
The campaign should feel like it belongs to the persona's cultural world, not like it is invading it.
Evaluate whether each concept feels native to this cultural context and score it in cultural_fit.

Audience precision lens:
Use audience_cares_about as a precision lens for question (1). Instead of generic attention, evaluate whether each concept addresses at least one explicit audience concern.
A concept can be creatively polished but still fail interest relevance if it answers the wrong audience question.

Input Schema:
- concept_portfolio: [object]
- platform_rules: object
- persona: object
- creative_brief.target_audience: string
- audience_cares_about: [string]

Output Requirements:
- reasoning: string
- concept_interest_scores: [
    {{
      "concept_id": string,
      "scroll_stop": boolean,
      "felt_emotion": boolean,
      "tell_a_friend": boolean,
      "cultural_fit": boolean,
      "overall_flag": "boring|standard|thumb_stopper"
    }}
  ]
- boring_flags: [string]
- thumb_stopper_flags: [string]

Return JSON that exactly matches the response schema and covers every concept in concept_portfolio.

INPUT JSON:
{json.dumps(prompt_payload, indent=2)}
"""

    invoke_start = time.time()
    logger.info(f"Agent [24]: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent [24]: Gemini Client instantiated. Sending prompt...")

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": InterestFilterAgentResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )

        api_duration = time.time() - invoke_start
        logger.info(f"Agent [24]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent [24]: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent [24]: Successfully parsed JSON response.")

        result = InterestFilterAgentResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent [24]: Successfully validated structured output with Pydantic.")
    except Exception as exc:
        logger.error("Agent [24]: Error during Gemini inference or JSON parsing for project_id=%s error=%s", project_id, exc)
        raise

    logger.info("Agent [24]: Updating PIPELINE collection...")
    try:
        total_duration = time.time() - start_time
        pipeline_log = {
            "agent_id": 24,
            "agent_name": "interest_filter_agent",
            "status": "completed",
            "timestamp": time.time(),
            "execution_time_sec": round(total_duration, 2),
            "duration_sec": round(total_duration, 2),
            "reasoning": result.reasoning,
            "output": {
                "concept_interest_scores": [row.model_dump() for row in result.concept_interest_scores],
                "boring_flags": result.boring_flags,
                "thumb_stopper_flags": result.thumb_stopper_flags,
            },
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )
        logger.info("Agent [24]: Successfully updated PIPELINE collection.")
    except Exception as exc:
        logger.error("Agent [24]: Error writing DB updates for project_id=%s error=%s", project_id, exc)
        raise

    total_duration = time.time() - start_time
    logger.info("Agent [24]: Execution completed in %.2fs for project_id=%s", total_duration, project_id)
    return result
