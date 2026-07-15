import json
import logging
import os
import time
from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.scene_role_enumeration_agent")

def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

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

class ViableRole(BaseModel):
    role: str = Field(description="Name of the narrative role evaluated")
    is_viable: bool = Field(description="Whether the role is viable for the scene elements")
    execution_direction: Optional[str] = Field(description="Sentence describing how execution would work. Required if viable.")
    why_not: Optional[str] = Field(description="Sentence stating why not. Required if not viable.")

class RoleEvaluation(BaseModel):
    role: str = Field(description="Name of the narrative role evaluated")
    atomic_fit_analysis: str = Field(description="Analysis of how atomic elements fit this role")
    viability_score: int = Field(description="Score out of 10 for viability")

class SceneRoleEnumerationResponse(BaseModel):
    viable_roles: List[ViableRole] = Field(description="Evaluation of viability for the seven narrative roles")
    reasoning: str = Field(description="Overall reasoning for the enumeration of roles")
    role_evaluation_table: List[RoleEvaluation] = Field(description="Detailed evaluation table used for pipeline logs")
    status: str = "completed"

async def run_scene_role_enumeration_agent(project_id: str, db) -> SceneRoleEnumerationResponse:
    """
    Agent 8: Scene Role Enumeration Agent
    Evaluates the scene's atomic elements against seven narrative roles.
    """
    agent_key = "scene_role_enumeration"
    logger.info(f"Initializing Agent [8]... | project_id={project_id}")
    
    try:
        # 1. Fetch data from collections
        logger.info(f"[{agent_key}] Fetching data for project_id={project_id}")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")
            
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        if not strategy_doc:
            raise ValueError(f"Strategy document for '{project_id}' not found")
            
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")

        # Defensive guard: narrative role enumeration is only valid for narrative-group formats.
        format_group = ideation_doc.get("format_group", "N")
        if format_group == "V":
            logger.info(f"[{agent_key}] Skipping — format_group is V (visual formats skip narrative role enumeration) | project_id={project_id}")
            result = SceneRoleEnumerationResponse(
                viable_roles=[],
                reasoning="Skipped — visual format group uses visual_structure_agent instead",
                role_evaluation_table=[],
                status="skipped",
            )
            return result

        # Extract values
        campaign_platform = strategy_doc.get("campaign_platform", "")
        human_truth = strategy_doc.get("human_truth", "")
        
        scene_intelligence = ideation_doc.get("scene_intelligence", {})
        atomic_elements = scene_intelligence.get("atomic_elements", {})
        narrative_budget = ideation_doc.get("narrative_budget", {})
        
        logger.info(f"[{agent_key}] Extracted inputs: atomic_elements, narrative_budget, campaign_platform, human_truth")

        # 2. Construct Prompt
        prompt = f"""
You are the Scene Role Enumeration Agent (Agent 8).
Your job is to enumerate every viable narrative role the preferred scene could play — preventing the most common failure mode in AI-generated ads, which is defaulting to a 'product demo in the middle'.

Evaluate whether the scene's atomic_elements could support each of these 7 narrative roles.
For viable roles, write one sentence describing how the execution would work.
For non-viable roles, state why not in one sentence.

Inputs:
- Scene Atomic Elements: {json.dumps(atomic_elements, indent=2)}
- Narrative Budget: {json.dumps(narrative_budget, indent=2)}
- Campaign Platform: {campaign_platform}
- Human Truth: {human_truth}

The seven narrative roles to evaluate:
1. Hook disruptor: The scene opens the ad and immediately creates pattern interrupt or emotional arrest. Best when the scene has high visual surprise or emotional specificity.
2. Conflict trigger: The scene creates or reveals the enemy — the problem the product will solve. Best when the scene has tension in the character's emotional state.
3. Proof demonstration: The scene shows the product working — tangible, specific, believable. Best when the product interaction type atomic element is strong.
4. Emotional climax: The scene is the peak emotional moment — the payoff before the offer. Best when the symbolic meaning atomic element is richest.
5. Transitional reveal: The scene marks a shift — before to after, dull to vivid, defeated to reclaimed. Best when the scene has inherent contrast energy.
6. Symbolic transformation moment: The scene operates as metaphor — the literal action carries the campaign’s deeper meaning. Best when symbolic meaning is layered.
7. Comedic inversion beat: The scene subverts expectations for tonal contrast or irony. Best in satire formats or when the scene can be read against its surface meaning.

Provide a complete response matching the required output JSON schema.
"""

        # 3. Call Gemini API
        invoke_start = time.time()
        logger.info(f"Agent [8]: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent [8]: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": SceneRoleEnumerationResponse.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent [8]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent [8]: Raw response length={len(response.text)} chars")

        # 4. Parse & Validate
        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent [8]: Successfully parsed JSON response.")

        result = SceneRoleEnumerationResponse(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent [8]: Successfully validated structured output with Pydantic.")

        # 5. DB Updates
        logger.info(f"Agent [8]: Updating IDEATION and PIPELINE collections...")
        
        # Write final output to IDEATION_COLLECTION using $set
        viable_roles_dicts = [role.dict() for role in result.viable_roles]
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "scene_intelligence.viable_roles": viable_roles_dicts
            }}
        )

        # Push to PIPELINE_COLLECTION using $push
        pipeline_log = {
            "agent_id": 8,
            "agent_name": "scene_role_enumeration_agent",
            "execution_time_utc": time.time(),
            "api_duration_s": round(api_duration, 2),
            "reasoning": result.reasoning,
            "role_evaluation_table": [role.dict() for role in result.role_evaluation_table]
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        logger.info(f"Agent [8]: Complete! Successfully updated MongoDB collections in {time.time() - invoke_start:.2f}s")
        return result

    except Exception as e:
        logger.error(f"[{agent_key}] Failed with error: {str(e)}", exc_info=True)
        raise
