import json
import logging
import os
import time

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.scene_role_selector_agent")

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

class SceneRoleSelectorResponse(BaseModel):
    selected_role: str = Field(description="The single optimal narrative role chosen from viable options.")
    reasoning: str = Field(description="Step by step reasoning evaluating the options against the four pressures.")
    role_selection_rationale: str = Field(description="A one-paragraph rationale for the selected role explaining the trade-off made — what other roles were considered and why this one was chosen.")
    trade_off_analysis: str = Field(description="Summary of trade-offs made against narrative duration, scroll behavior, momentum, and SMP.")
    status: str = "completed"

async def run_scene_role_selector_agent(project_id: str, db) -> SceneRoleSelectorResponse:
    """
    Agent 9: Scene Role Selector Agent
    Selects the single optimal narrative role from viable options.
    Balances duration cost, narrative momentum, platform scroll behavior, and SMP alignment.
    """
    agent_key = "scene_role_selector"
    logger.info(f"Initializing Agent [9]... | project_id={project_id}")
    
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

        # Extract values
        # Strategy
        strategy_agents = strategy_doc.get("agents", {})
        strategy_models = strategy_agents.get("strategy_models", {})
        creative_brief = strategy_models.get("creative_brief", {})
        single_minded_proposition = creative_brief.get("single_minded_proposition", "")
        if not single_minded_proposition and isinstance(strategy_models, dict):
            # Fallback for alternative structures
            single_minded_proposition = strategy_models.get("single_minded_proposition", "")
            
        audience_persona = strategy_agents.get("audience_persona", {})
        media_habits = audience_persona.get("media_habits", [])
        
        # Ideation
        scene_intelligence = ideation_doc.get("scene_intelligence", {})
        viable_roles = scene_intelligence.get("viable_roles", [])
        narrative_budget = ideation_doc.get("narrative_budget", {})
        video_type_final = ideation_doc.get("video_type_final", "Unknown")
        
        logger.info(f"[{agent_key}] Extracted inputs: viable_roles ({len(viable_roles)}), narrative_budget, video_type_final, single_minded_proposition, media_habits")

        # 2. Construct Prompt
        prompt = f"""
You are the Scene Role Selector Agent (Agent 9).
Your job is to select the single optimal narrative role for the preferred scene from the viable_roles enumerated. 

The selection must balance four competing pressures:
1. Duration pressure: In short formats, the role must justify its time cost. A symbolic transformation moment that requires 5 seconds to land is unsuitable for a 15-second ad.
2. Narrative momentum: The selected role must strengthen the story arc, not interrupt it. A scene placed as hook disruptor in a narrative archetype built for slow emotional escalation creates tonal dissonance.
3. Platform scroll behaviour: On high-velocity feeds, hook or conflict trigger roles win. On in-stream or longer formats, emotional climax or symbolic roles can be sustained.
4. Strategic emphasis: Which role most directly reinforces the single_minded_proposition? The role that best amplifies the SMP wins all else equal.

Produce a one-paragraph rationale for the selected role explaining the trade-off made — what other roles were considered and why this one was chosen.

Inputs:
- Viable Roles: {json.dumps(viable_roles, indent=2)}
- Narrative Budget: {json.dumps(narrative_budget, indent=2)}
- Video Type (Format): {video_type_final}
- Single Minded Proposition: {single_minded_proposition}
- Media Habits (Platform contexts): {json.dumps(media_habits, indent=2)}

Provide a complete response matching the required output JSON schema.
"""

        # 3. Call Gemini API
        invoke_start = time.time()
        logger.info(f"Agent [9]: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent [9]: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": SceneRoleSelectorResponse.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent [9]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent [9]: Raw response length={len(response.text)} chars")

        # 4. Parse & Validate
        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent [9]: Successfully parsed JSON response.")

        result = SceneRoleSelectorResponse(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent [9]: Successfully validated structured output with Pydantic.")

        # 5. DB Updates
        logger.info(f"Agent [9]: Updating IDEATION and PIPELINE collections...")
        
        # Write final output to IDEATION_COLLECTION using $set
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "scene_intelligence.selected_role": result.selected_role
            }}
        )

        # Push to PIPELINE_COLLECTION using $push
        pipeline_log = {
            "agent_id": 9,
            "agent_name": "scene_role_selector_agent",
            "execution_time_utc": time.time(),
            "api_duration_s": round(api_duration, 2),
            "reasoning": result.reasoning,
            "role_selection_rationale": result.role_selection_rationale,
            "trade_off_analysis": result.trade_off_analysis,
            "selected_role": result.selected_role
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        logger.info(f"Agent [9]: Complete! Successfully updated MongoDB collections in {time.time() - invoke_start:.2f}s")
        return result

    except Exception as e:
        logger.error(f"[{agent_key}] Failed with error: {str(e)}", exc_info=True)
        raise
