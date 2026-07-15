import json
import logging
import os
import time
from typing import List, Optional, Literal

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.constraint_priority_resolver")

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")

# ── Pydantic Models for Structured Output ────────────────────────────────────

class ResolvedConflict(BaseModel):
    conflict: str = Field(description="Description of the constraint conflict")
    resolution: str = Field(description="How the conflict was resolved")
    tier_applied: Literal["strategic_importance", "user_priority", "platform_survivability"] = Field(description="Which tier of the hierarchy was applied to resolve this")
    trade_off_cost: str = Field(description="Theoretical cost of this trade-off for the campaign")

class PriorityDirectives(BaseModel):
    resolved_conflicts: List[ResolvedConflict]
    operating_mode: Literal["conservative", "experimental"] = Field(description="The operating mode that guided the resolutions")

class ConflictResolutionLogEntry(BaseModel):
    conflict: str
    tier_1_evaluation: str
    tier_2_evaluation: str
    tier_3_evaluation: str
    winning_tier: str
    final_resolution: str

class ConstraintPriorityResolverResult(BaseModel):
    priority_directives: PriorityDirectives
    reasoning: str = Field(description="Overall reasoning for constraint resolution strategy")
    full_conflict_resolution_log: List[ConflictResolutionLogEntry] = Field(description="Detailed resolution logs per conflict")
    operating_mode_rationale: str = Field(description="Why this operating mode was selected or maintained")


def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

async def run_constraint_priority_resolver_agent(project_id: str, db) -> ConstraintPriorityResolverResult:
    """
    Agent 2: Constraint Priority Resolver
    Resolves constraint conflicts flagged by Agent 1 using a three-tier hierarchy.
    Emits structured directives for downstream agents.
    """
    agent_key = "constraint_priority_resolver"
    logger.info(f"[{agent_key}] Starting agent execution | project_id={project_id}")

    try:
        # 1. Fetch data from collections
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")

        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": project_id})
        if not strategy_doc:
            raise ValueError(f"Strategy for project '{project_id}' not found")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": project_id})
        if not ideation_doc:
            raise ValueError(f"Ideation (Agent 1 constraints) for project '{project_id}' not found")

        # 2. Extract specific inputs required
        video_length_seconds = project_doc.get("video_length_seconds")
        operating_mode = project_doc.get("operating_mode", "conservative") # Defaults to conservative
        
        # From strategy
        # We need the single minded proposition
        strategy_agents = strategy_doc.get("agents", {})
        smp = strategy_agents.get("strategy_models", {}).get("creative_brief", {}).get("single_minded_proposition", "")
        mandatories = strategy_agents.get("strategy_models", {}).get("creative_brief", {}).get("mandatories", [])

        # From ideation
        constraint_graph = ideation_doc.get("constraint_graph", {})
        hard_constraints = constraint_graph.get("hard_constraints", [])
        soft_constraints = constraint_graph.get("soft_constraints", [])
        feasibility_envelope = constraint_graph.get("feasibility_envelope", "")
        conflict_flags = constraint_graph.get("conflict_flags", [])

        if not conflict_flags:
            logger.info(f"[{agent_key}] No explicit conflict flags found from Agent 1. Attempting resolution across hard constraints implicitly if any.")

        # 3. Assemble prompt
        prompt = f"""
        You are the Constraint Priority Resolver (Agent 2).
        Your job is to resolve all constraint conflicts (flagged or implicit in hard constraints) using a strict three-tier priority hierarchy.

        --- Inputs ---
        Ideation (from Agent 1):
        hard_constraints: {json.dumps(hard_constraints)}
        soft_constraints: {json.dumps(soft_constraints)}
        feasibility_envelope: {feasibility_envelope}
        conflict_flags: {json.dumps(conflict_flags)}

        Strategy:
        single_minded_proposition: "{smp}"
        mandatories: {json.dumps(mandatories)}

        Projects:
        video_length_seconds: {video_length_seconds}
        operating_mode (default): {operating_mode}

        --- Task ---
        Apply the following three-tier priority hierarchy to resolve constraint conflicts:

        Tier 1 — Strategic importance:
        Does satisfying this constraint more directly serve the single_minded_proposition? The constraint that better serves SMP wins.

        Tier 2 — Explicit user priority:
        Did the brand explicitly mark this as required (e.g., via mandatories)? Explicit user intent overrides implied preferences.

        Tier 3 — Platform survivability:
        Which constraint resolution produces a concept more likely to survive its platform's attention dynamics? The more platform-compatible resolution wins.

        Evaluate every conflict through these three tiers to log the `full_conflict_resolution_log`.
        For every conflict resolved, emit a structured directive (`resolved_conflicts`) detailing what was compressed/preserved and the trade-off cost.
        Confirm the final `operating_mode` (conservative vs experimental) that fits the resolutions. Conservative biases toward brand safety; Experimental biases toward creative audacity.

        Return strictly in the required JSON format matching the schema.
        """

        # 4. Invoke LLM
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        
        logger.info(f"[{agent_key}] Sending request to Gemini...")
        api_start_time = time.time()
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ConstraintPriorityResolverResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - api_start_time
        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")

        clean_text = _clean_json_string(response.text)
        result_data = json.loads(clean_text)
        structured_result = ConstraintPriorityResolverResult(**result_data)

        # 5. Save to MongoDB
        priority_directives_data = structured_result.priority_directives.model_dump()
        
        # Update ideation collection
        from datetime import datetime, timezone
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": project_id},
            {"$set": {
                "priority_directives": priority_directives_data,
                f"status.{agent_key}": "completed"
            }},
            upsert=True
        )

        pipeline_log_entry = {
            "agent_key": agent_key,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc),
            "reasoning": structured_result.reasoning,
            "full_conflict_resolution_log": [log.model_dump() for log in structured_result.full_conflict_resolution_log],
            "operating_mode_rationale": structured_result.operating_mode_rationale
        }

        # Update pipeline collection
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": project_id},
            {"$push": {"agent_logs": pipeline_log_entry}},
            upsert=True
        )

        logger.info(f"[{agent_key}] Completed agent execution | project_id={project_id}")
        return structured_result

    except Exception as e:
        logger.error(f"[{agent_key}] Error during execution: {e}")
        from datetime import datetime, timezone
        
        # Mark as failed in DB
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": project_id},
            {"$set": {f"status.{agent_key}": "error"}},
            upsert=True
        )
        
        failure_log_entry = {
            "agent_key": agent_key,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc),
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": project_id},
            {"$push": {"agent_logs": failure_log_entry}},
            upsert=True
        )
        raise e
