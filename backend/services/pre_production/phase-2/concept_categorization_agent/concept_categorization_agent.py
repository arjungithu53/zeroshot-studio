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

logger = logging.getLogger("zeroshot.phase2.concept_categorization_agent")


class ConceptCategoryUpdate(BaseModel):
    concept_id: str = Field(description="Concept identifier.")
    category: Literal["PITCH", "PLAY", "PLUNGE"] = Field(description="Assigned concept category.")


class CategoryAssignmentLogEntry(BaseModel):
    concept_id: str = Field(description="Concept identifier.")
    category: Literal["PITCH", "PLAY", "PLUNGE"] = Field(description="Assigned concept category.")
    rationale: str = Field(description="Reasoning for category assignment.")


class PortfolioBalanceAudit(BaseModel):
    pitch_count: int = Field(description="Number of concepts assigned to PITCH.")
    play_count: int = Field(description="Number of concepts assigned to PLAY.")
    plunge_count: int = Field(description="Number of concepts assigned to PLUNGE.")
    is_balanced: bool = Field(description="Whether portfolio meets minimum distribution rule.")
    rebalance_flag: Optional[str] = Field(default=None, description="Rebalance instruction or null.")


class ConceptCategorizationResult(BaseModel):
    status: Literal["completed", "skipped", "error"] = Field(default="completed")
    reason: Optional[str] = Field(default=None, description="Reason when skipped or errored.")
    concept_portfolio_updates: List[ConceptCategoryUpdate] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Reasoning behind category portfolio composition.")
    category_assignment_log: List[CategoryAssignmentLogEntry] = Field(default_factory=list)
    portfolio_balance_audit: PortfolioBalanceAudit = Field(
        default_factory=lambda: PortfolioBalanceAudit(
            pitch_count=0,
            play_count=0,
            plunge_count=0,
            is_balanced=False,
            rebalance_flag="insufficient_assignments",
        )
    )


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
    updates: List[ConceptCategoryUpdate],
) -> List[Dict[str, Any]]:
    updates_by_id = {u.concept_id: u for u in updates}
    merged: List[Dict[str, Any]] = []

    for index, concept in enumerate(concept_portfolio):
        concept_item = dict(concept)
        concept_id = str(concept_item.get("concept_id", "")).strip()
        selected_update = updates_by_id.get(concept_id)

        if selected_update is None and updates:
            logger.warning(
                "Agent 22: Missing concept_id match for portfolio index=%s (concept_id=%s). Falling back by index.",
                index,
                concept_id,
            )
            if index < len(updates):
                selected_update = updates[index]

        if selected_update is not None:
            concept_item["category"] = selected_update.category

        merged.append(concept_item)

    return merged


async def run_concept_categorization_agent(project_id: str, db: Any) -> ConceptCategorizationResult:
    logger.info("Initializing Agent 22 (Concept Categorization Agent) for project_id=%s", project_id)
    start_time = time.time()

    try:
        project_obj_id = ObjectId(project_id)
    except Exception as exc:
        logger.error("Agent 22: Invalid project_id format for ObjectId conversion. project_id=%s error=%s", project_id, exc)
        raise ValueError(f"Invalid project_id {project_id}")

    logger.info("Agent 22: Fetching data for project_id=%s", project_id)
    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": project_obj_id})
    if not project_doc:
        logger.error("Agent 22: Project document not found in %s for project_id=%s", PROJECTS_COLLECTION, project_id)
        raise ValueError(f"Project not found in {PROJECTS_COLLECTION}")

    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}

    concept_portfolio = ideation_doc.get("concept_portfolio", [])
    platform_rules = ideation_doc.get("platform_rules", {})
    video_length_seconds = project_doc.get("video_length_seconds")
    selected_archetype = ideation_doc.get("selected_archetype", {})
    narrative_budget = ideation_doc.get("narrative_budget", {})
    strategy_agents = strategy_doc.get("agents", {})

    logger.info("Agent 22: Extracted key inputs from DB docs.")
    logger.info(
        "Agent 22: Input summary | concept_count=%s platform_rules_present=%s video_length_seconds=%s selected_archetype_present=%s narrative_budget_present=%s strategy_agents_present=%s",
        len(concept_portfolio) if isinstance(concept_portfolio, list) else 0,
        isinstance(platform_rules, dict) and bool(platform_rules),
        video_length_seconds,
        isinstance(selected_archetype, dict) and bool(selected_archetype),
        isinstance(narrative_budget, dict) and bool(narrative_budget),
        isinstance(strategy_agents, dict) and bool(strategy_agents),
    )

    if not isinstance(concept_portfolio, list) or not concept_portfolio:
        logger.error("Agent 22: concept_portfolio missing or empty in %s for project_id=%s", IDEATION_COLLECTION, project_id)
        raise ValueError("concept_portfolio is missing in ideation document")

    prompt_payload = {
        "concept_portfolio": concept_portfolio,
        "platform_rules": platform_rules,
        "video_length_seconds": video_length_seconds,
        "selected_archetype": selected_archetype,
        "narrative_budget": narrative_budget,
        "strategy_agents": strategy_agents,
    }

    prompt = f"""
You are Agent 22: concept_categorization_agent.

RUN CONDITION: ALWAYS

Source framework:
- Ch.12 'Pitch Play Plunge'

Purpose - Grounded in Pitch/Play/Plunge Framework:
Assign each concept to exactly one category:
- PITCH (immediate consumption, 3-15s, feed, fully intelligible sound-off)
- PLAY (interactive/participatory, social action built in, suitable for Stories/WhatsApp)
- PLUNGE (immersive, 20+s, emotional depth, in-stream attention)

Audit portfolio balance rule:
A balanced concept portfolio requires minimum 3 PITCH, 1-2 PLAY, and 1 PLUNGE.
If all concepts are PITCH or the minimum mix is not satisfied, set rebalance_flag to a concise remediation note.

Input Schema:
- concept_portfolio: [object]
- platform_rules: object
- video_length_seconds: int

Output Requirements:
1) concept_portfolio_updates:
   - Per concept: {{ "concept_id": string, "category": "PITCH|PLAY|PLUNGE" }}
2) reasoning: string
3) category_assignment_log:
   - List entries: {{ "concept_id": string, "category": string, "rationale": string }}
4) portfolio_balance_audit:
   - {{
       "pitch_count": int,
       "play_count": int,
       "plunge_count": int,
       "is_balanced": boolean,
       "rebalance_flag": string|null
     }}

Prompt Logic:
Step 1: For each concept, read concept_hook, story_beats, virality_lever, offer_placement, and cta_framing.
Step 2: Categorize each concept using strict criteria:
- PITCH:
  Immediate consumption. Hook lands in under 3 seconds. Entire message can be processed in one rapid sequence.
  Suitable for feed placements and 3-15 second runtime. Must remain intelligible on sound-off.
- PLAY:
  Interactive or participatory. Concept invites explicit response: reaction, share, tag, challenge, split purchase, poll, or social handoff.
  Participation cue is built into narrative design.
- PLUNGE:
  Immersive long-form. Concept earns attention through emotional depth and layered payoff.
  Designed for 20+ seconds and intentional viewing.
Step 3: Ensure each concept has exactly one category assignment and provide rationale.
Step 4: Audit portfolio distribution and compute balance fields exactly.

Return JSON that exactly matches the response schema and includes all concepts from concept_portfolio.

INPUT JSON:
{json.dumps(prompt_payload, indent=2)}
"""

    invoke_start = time.time()
    logger.info(f"Agent 22: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 22: Gemini Client instantiated. Sending prompt...")

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ConceptCategorizationResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )

        api_duration = time.time() - invoke_start
        logger.info(f"Agent 22: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 22: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 22: Successfully parsed JSON response.")

        result = ConceptCategorizationResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 22: Successfully validated structured output with Pydantic.")
    except Exception as exc:
        logger.error("Agent 22: Error during Gemini inference or JSON parsing for project_id=%s error=%s", project_id, exc)
        raise

    merged_portfolio = _apply_updates_to_portfolio(concept_portfolio, result.concept_portfolio_updates)

    logger.info("Agent 22: Updating IDEATION and PIPELINE collections...")
    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {
                "$set": {
                    "concept_portfolio": merged_portfolio,
                    "status.concept_categorization_agent": "completed",
                    "updated_at": time.time(),
                }
            },
            upsert=True,
        )

        total_duration = time.time() - start_time
        pipeline_log = {
            "agent_id": 22,
            "agent_name": "concept_categorization_agent",
            "status": "completed",
            "timestamp": time.time(),
            "execution_time_sec": round(total_duration, 2),
            "duration_sec": round(total_duration, 2),
            "reasoning": result.reasoning,
            "output": {
                "concept_portfolio_updates": [u.model_dump() for u in result.concept_portfolio_updates],
                "category_assignment_log": [entry.model_dump() for entry in result.category_assignment_log],
                "portfolio_balance_audit": result.portfolio_balance_audit.model_dump(),
            },
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )
        logger.info("Agent 22: Successfully updated IDEATION and PIPELINE collections.")
    except Exception as exc:
        logger.error("Agent 22: Error writing DB updates for project_id=%s error=%s", project_id, exc)
        raise

    total_duration = time.time() - start_time
    logger.info("Agent 22: Execution completed in %.2fs for project_id=%s", total_duration, project_id)
    return result
