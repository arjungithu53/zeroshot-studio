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
logger = logging.getLogger("zeroshot.idea_core_preservation")

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

class IdeaCoreRules(BaseModel):
    status: Literal["completed", "skipped"]
    reason: Optional[str] = None
    idea_structural_role: Optional[Literal["hook_driver", "conflict_trigger", "symbolic_device", "payoff_resolution"]] = None
    what_counts_as_integration: Optional[str] = None
    what_counts_as_superficial: Optional[str] = None
    integrity_tests: Optional[List[str]] = None

class IdeaCorePreservationResult(BaseModel):
    idea_core_rules: IdeaCoreRules
    reasoning: Optional[str] = Field(None, description="Reasoning for rule generation")
    structural_role_analysis: Optional[str] = Field(None, description="Analysis for structural role")
    integrity_test_rationale: Optional[List[str]] = Field(None, description="Rationale for tests")


def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

async def run_idea_core_preservation_agent(project_id: str, db) -> IdeaCorePreservationResult:
    """
    Agent 3: Idea Core Preservation
    Defines structural integrity rules all concept generation agents must follow.
    Activates only when a brand idea is provided (project.idea is NOT null).
    """
    agent_key = "idea_core_preservation"
    logger.info(f"[{agent_key}] Starting agent execution | project_id={project_id}")

    try:
        # 1. Fetch project data to check the RUN CONDITION
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")

        idea = project_doc.get("idea")

        # Check RUN CONDITION: IF project.idea is NOT null | SKIP if idea is null
        if not idea or idea.strip() == "":
            logger.info(f"[{agent_key}] project.idea is null or empty. Skipping agent execution.")
            
            skipped_rules = IdeaCoreRules(
                status="skipped",
                reason="project.idea is null - no brand idea provided"
            )
            
            skipped_result = IdeaCorePreservationResult(
                idea_core_rules=skipped_rules
            )
            
            # Save skipped state to IDEATION
            from datetime import datetime, timezone
            await db[IDEATION_COLLECTION].update_one(
                {"project_id": project_id},
                {"$set": {
                    "idea_core_rules": skipped_rules.model_dump(),
                    f"status.{agent_key}": "skipped"
                }},
                upsert=True
            )
            
            # Update pipeline
            pipeline_log_entry = {
                "agent_key": agent_key,
                "status": "skipped",
                "timestamp": datetime.now(timezone.utc),
                "reason": "project.idea is null - no brand idea provided"
            }
            await db[PIPELINE_COLLECTION].update_one(
                {"project_id": project_id},
                {"$push": {"agent_logs": pipeline_log_entry}},
                upsert=True
            )
            
            return skipped_result


        # 2. Fetch strategy and ideation data since we are proceeding
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": project_id})
        if not strategy_doc:
            raise ValueError(f"Strategy for project '{project_id}' not found")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": project_id})
        if not ideation_doc:
            raise ValueError(f"Ideation for project '{project_id}' not found")

        # 3. Extract specific inputs required
        strategy_agents = strategy_doc.get("agents", {})
        campaign_platform = strategy_agents.get("truth_conflict_platform", {}).get("strategy_models", {}).get("campaign_platform", "")
        # Try finding single_minded_proposition
        smp = strategy_agents.get("strategy_models", {}).get("creative_brief", {}).get("single_minded_proposition", "")
        
        if not campaign_platform:
            # Fallback based on potential schema paths from strategy
            campaign_platform = strategy_agents.get("truth_conflict_platform", {}).get("platform_narrative", "")
            
        constraint_graph = ideation_doc.get("constraint_graph", {})
        idea_classification = constraint_graph.get("idea_classification", "")

        # 4. Assemble prompt
        prompt = f"""
        You are the Idea Core Preservation (Agent 3).
        Your job is to define structural integrity rules for downstream concept generation agents to follow.
        The goal is to prevent the LLM's tendency to politely reference a brand idea superficially but drift away from it structurally.

        --- Inputs ---
        Projects:
        idea: "{idea}"

        Ideation (from Agent 1):
        constraint_graph.idea_classification: "{idea_classification}"

        Strategy:
        campaign_platform: "{campaign_platform}"
        creative_brief.single_minded_proposition: "{smp}"

        --- Task ---
        Complete the following steps to build the `idea_core_rules` and pipeline logs.

        Step 1 — Structural Role
        Using constraint_graph.idea_classification, determine the structural role the idea must play.
        Categorize it into one of the following: 'hook_driver', 'conflict_trigger', 'symbolic_device', 'payoff_resolution'.
        - hook_driver: concept opens with a direct execution/evolution of the idea.
        - conflict_trigger: the idea creates narrative tension.
        - symbolic_device: the idea appears as a recognizable visual or narrative motif.
        - payoff_resolution: the idea must close every concept.
        Log your analysis in `structural_role_analysis`.

        Step 2 — Define integration vs superficial reference
        Define what constitutes a superficial reference vs. genuine structural integration for this specific idea.
        - A superficial reference is flavouring (a word in the hook, a background visual element).
        - Genuine integration means the idea changes the narrative architecture — remove it and the concept no longer works.
        Write clear rules in `what_counts_as_integration` and `what_counts_as_superficial`.

        Step 3 — Write Integrity Tests
        Write three explicit, testable questions that the Concept Kill Switch agent will apply to each concept:
        1. Is the brand idea structurally load-bearing in this concept?
        2. Would removing the idea require restructuring at least one story beat?
        3. Does the concept's emotional payoff depend on the idea being present?
        Tailor these general questions to the specific idea provided. Provide your rationale in `integrity_test_rationale`.

        Set the `status` of idea_core_rules to "completed", and `reason` to null.
        Provide overall `reasoning`.

        Return STRICTLY in the requested JSON format matching the schema.
        """

        # 5. Invoke LLM
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
                "response_json_schema": IdeaCorePreservationResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - api_start_time
        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")

        clean_text = _clean_json_string(response.text)
        result_data = json.loads(clean_text)
        structured_result = IdeaCorePreservationResult(**result_data)

        # 6. Save to MongoDB
        idea_core_rules_data = structured_result.idea_core_rules.model_dump()
        
        from datetime import datetime, timezone
        
        # Update ideation collection
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": project_id},
            {"$set": {
                "idea_core_rules": idea_core_rules_data,
                f"status.{agent_key}": "completed"
            }},
            upsert=True
        )

        pipeline_log_entry = {
            "agent_key": agent_key,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc),
            "reasoning": structured_result.reasoning,
            "structural_role_analysis": structured_result.structural_role_analysis,
            "integrity_test_rationale": structured_result.integrity_test_rationale
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
