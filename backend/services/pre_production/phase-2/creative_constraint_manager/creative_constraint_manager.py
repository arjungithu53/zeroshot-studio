import json
import logging
import os
import time
from typing import List, Optional, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.creative_constraint_manager")

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

class ConflictLogEntry(BaseModel):
    constraint_a: str
    constraint_b: str
    conflict_type: str
    flagged_for_resolver: bool

class CreativeConstraintManagerResult(BaseModel):
    # Log fields
    reasoning: str = Field(description="Explanation of the constraint classification and feasibility envelope")
    constraint_conflict_log: List[ConflictLogEntry] = Field(description="List of constraint conflicts")
    idea_classification_detail: Optional[str] = Field(description="Details on why the idea was classified as such")
    
    # constraint_graph
    hard_constraints: List[str] = Field(description="Non-negotiable constraints like exact duration, mandatory scenes")
    soft_constraints: List[str] = Field(description="Strongly preferred but negotiable under conflict (tone, format)")
    optional_constraints: List[str] = Field(description="Nice-to-have constraints that enrich but do not define")
    feasibility_envelope: str = Field(description="Maximum narrative complexity this duration can support")
    idea_classification: Optional[Literal["hook_driver", "conflict_trigger", "symbolic_device", "payoff_resolution"]] = Field(description="Classification of the brand-supplied idea")
    video_type_options: Optional[List[str]] = Field(description="Compatible video types if video type is TBD")
    conflict_flags: List[str] = Field(description="Notes on conflicting constraints flagged for Agent 2")


def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

async def run_creative_constraint_manager_agent(project_id: str, db) -> CreativeConstraintManagerResult:
    """
    Agent 1: Creative Constraint Manager
    Ingests all inputs from strategy.agents.* (mapped into state) and projects.
    Classifies every input into hard/soft/optional constraint buckets. Computes the feasibility envelope.
    Classifies brand idea structural role if present. Flags whether video type is confirmed or TBD.
    """
    agent_key = "creative_constraint_manager"
    logger.info(f"[{agent_key}] Starting agent execution | project_id={project_id}")

    try:
        # 1. Fetch project data (using ObjectId for the projects collection)
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            logger.error(f"[{agent_key}] Project not found | project_id={project_id}")
            raise ValueError(f"Project '{project_id}' not found")

        # 2. Fetch strategy data (using string project_id for the strategy collection)
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": project_id})
        if not strategy_doc:
            logger.error(f"[{agent_key}] Strategy document not found | project_id={project_id}")
            raise ValueError(f"Strategy for project '{project_id}' not found")

        # 3. Assemble inputs (simulate the schema required by prompt)
        # Extract from projects
        video_length_seconds = project_doc.get("video_length_seconds")
        video_type = project_doc.get("video_type", "TBD")
        preferred_scene = project_doc.get("preferred_scene")
        idea = project_doc.get("idea")
        product_details = project_doc.get("product_details", "")
        target_audience = project_doc.get("target_audience", {})
        brand_guidelines = project_doc.get("brand_guidelines")

        # Extract from strategy
        # Typically these are nested in the agents' outputs. Let's send the whole strategy to LLM or mapped ones.
        strategy_json = json.dumps(strategy_doc.get("agents", {}), default=str)

        prompt = f"""
        You are the Entry Orchestration Agent (Agent 1: Creative Constraint Manager).
        Your job is to convert all structured and unstructured inputs into a normalized constraint graph that every downstream agent reasons over.
        You must be exhaustive and precise — downstream agents cannot negotiate constraints this agent fails to classify.

        --- Inputs ---
        Projects:
        video_length_seconds: {video_length_seconds}
        video_type: {video_type}
        preferred_scene: {preferred_scene}
        idea: {idea}
        product_details: {product_details}
        target_audience: {target_audience}
        brand_guidelines: {brand_guidelines}

        Strategy Outputs (from phase 1):
        {strategy_json}

        --- Task ---
        Step 1 — Ingest all available inputs and classify every constraint into one of three categories:
        - Hard constraints: non-negotiable (e.g., video_length_seconds, preferred_scene, mandatories, explicit 'must include' from idea)
        - Soft constraints: strongly preferred but negotiable under conflict (e.g., tone of voice, offer format, support points)
        - Optional constraints: nice-to-have (e.g., persona media habits, visual context color palette)

        Step 2 — Assign priority weights to all hard constraints.
        If there are conflicts between hard constraints, produce a degradation plan in the reasoning mapping which elements compress. Log these in conflict_log.

        Step 3 — Compute the feasibility envelope.
        Based on video_length_seconds, determine maximum narrative complexity. Express as a plain English rule (e.g. "At 32 seconds: supports one tension arc with product demo..."). This is a ceiling, not a target.

        Step 4 — Idea classification.
        If idea is provided, classify it as: "hook_driver", "conflict_trigger", "symbolic_device", or "payoff_resolution". Provide detail.

        Step 5 — Video Type selection.
        If video_type is "TBD", log which video type options are compatible with the constraint envelope.

        Return strictly in the required JSON format.
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
                "response_json_schema": CreativeConstraintManagerResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - api_start_time
        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")

        clean_text = _clean_json_string(response.text)
        result_data = json.loads(clean_text)
        structured_result = CreativeConstraintManagerResult(**result_data)

        # 5. Save to MongoDB
        constraint_graph = {
            "hard_constraints": structured_result.hard_constraints,
            "soft_constraints": structured_result.soft_constraints,
            "optional_constraints": structured_result.optional_constraints,
            "feasibility_envelope": structured_result.feasibility_envelope,
            "idea_classification": structured_result.idea_classification,
            "video_type_options": structured_result.video_type_options,
            "conflict_flags": structured_result.conflict_flags,
        }
        
        pipeline_log = {
            "reasoning": structured_result.reasoning,
            "constraint_conflict_log": [log.model_dump() for log in structured_result.constraint_conflict_log],
            "idea_classification_detail": structured_result.idea_classification_detail
        }

        # Update ideation collection
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": project_id},
            {"$set": {
                "constraint_graph": constraint_graph,
                f"status.{agent_key}": "completed"
            }},
            upsert=True
        )

        from datetime import datetime, timezone
        pipeline_log_entry = {
            "agent_key": agent_key,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc),
            "reasoning": structured_result.reasoning,
            "constraint_conflict_log": [log.model_dump() for log in structured_result.constraint_conflict_log],
            "idea_classification_detail": structured_result.idea_classification_detail
        }

        # Update pipeline collection (push into agent_logs array as done in phase 1)
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
