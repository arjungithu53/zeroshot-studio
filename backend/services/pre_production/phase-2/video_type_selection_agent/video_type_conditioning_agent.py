import json
import logging
import os
import time
import asyncio
from typing import List, Optional, Literal, Dict, Any

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.video_type_conditioning")

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL" , "gemini-3.1-pro-preview") # Or configured model
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")

# ── Pydantic Models for Structured Output ────────────────────────────────────

class FormatPhysicsRules(BaseModel):
    hook_pacing: str = Field(description="Rules regarding the hook pacing for this format.")
    authenticity_signal: str = Field(description="Requirements for authenticity signals.")
    metaphor_tolerance: str = Field(description="Tolerance for metaphor exaggeration.")
    fourth_wall: bool = Field(description="Whether fourth-wall breaks are enabled.")
    proof_emphasis: str = Field(description="Emphasis on proof elements.")

class VideoTypeConditioningResult(BaseModel):
    status: Literal["completed", "error", "skipped"]
    reason: Optional[str] = None
    
    # Ideation Fields
    video_type_final: Optional[str] = Field(None, description="The confirmed video format")
    video_type_conditioning_notes: Optional[str] = Field(None, description="A 3-sentence conditioning directive")
    
    # Pipeline Log Fields
    reasoning: Optional[str] = Field(None, description="Full reasoning for the applied conditioning")
    format_physics_rules: Optional[FormatPhysicsRules] = None
    suppressed_directions: Optional[List[str]] = Field(None, description="Creative directions to suppress for this format")

# ---------------------------------------------------------------------------
# Core Agent Function
# ---------------------------------------------------------------------------

def _clean_json_string(raw_response: str) -> str:
    """Removes markdown code blocks to safely parse JSON."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

async def run_video_type_conditioning_agent(project_id: str, db) -> VideoTypeConditioningResult:
    """
    Agent 5a: Video Type Conditioning.
    When video type is confirmed, conditions the creative search space for that format.
    Modifies permissible narrative grammar, hook pacing, authenticity signal requirements, and metaphor exaggeration tolerance.
    RUN CONDITION: Skip if projects.video_type IS 'TBD'.
    """
    logger.info("Initializing Agent 5a (Video Type Conditioning) | project_id=%s", project_id)
    start_time = time.time()

    # 1. Fetch relevant data
    try:
        proj_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not proj_doc:
            raise ValueError(f"Project not found: {project_id}")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}
        
    except Exception as e:
        logger.error("Failed to fetch project, ideation or strategy docs  |  error=%s", e)
        return VideoTypeConditioningResult(status="error", reason=str(e))

    # Check rule
    video_type = proj_doc.get("video_type", "TBD")
    if video_type.upper() == "TBD" or not video_type:
        logger.info("Skipping Agent 5a: video_type is 'TBD' | project_id=%s", project_id)
        return VideoTypeConditioningResult(status="skipped", reason="video_type is TBD")

    # Extract required inputs
    video_length_seconds = proj_doc.get("video_length_seconds")
    # From ideation
    constraint_graph = ideation_doc.get("constraint_graph", {})
    feasibility_envelope = constraint_graph.get("feasibility_envelope", "")
    priority_directives = ideation_doc.get("priority_directives", {})
    enemy_type = ideation_doc.get("enemy_type", "")  # Or strategy.agents.conflict_identification.enemy
    if not enemy_type:
        conflict_identification = strategy_doc.get("agents", {}).get("conflict_identification", {})
        enemy_type = conflict_identification.get("enemy", "Unknown Enemy")

    # 2. Construct prompt
    prompt = f"""
    You are the Video Type Conditioning Agent (Agent 5a) for an advertising campaign.
    
    PURPOSE:
    When video type is confirmed (not "TBD"), you condition the creative search space for that format. 
    You will modify permissible narrative grammar, hook pacing, authenticity signal requirements, and 
    metaphor exaggeration tolerance based on the format provided.
    
    INPUT DATA:
    Project Context:
      video_type: "{video_type}"
      video_length_seconds: "{video_length_seconds}"
    
    Ideation Context:
      constraint_graph.feasibility_envelope: "{feasibility_envelope}"
      priority_directives: {json.dumps(priority_directives) if priority_directives else "{}"}
      enemy_type: "{enemy_type}"
      
    FORMAT CONDITIONING RULES BY VIDEO TYPE:
    - UGC / Organic-style: Increase authenticity signal requirements. Disable highly produced metaphor sequences. Enable fourth-wall breaks. Hook must feel unscripted — avoid branded opening cards. Persona must feel like the creator, not the subject.
    - Testimonial / Real person: Increase proof emphasis. Require at least one specific result claim in story beats. Emotional sincerity must be primary — no irony or distance. Product interaction must be shown, not implied.
    - Animation / Illustrated: Expand metaphor exaggeration tolerance. Enable abstract visual sequences impractical in live action. Allow symbolic product transformations. Tone can heighten to epic or surreal.
    - Narrative / Mini-film: Enable multi-character arcs. Allow slower hook pacing — up to 4 seconds before brand mention. Emotional escalation can be gradual. Offer communication must be compressed into the final beat.
    - Satire / Comedy: Increase tolerance for absurd pattern interrupts. The enemy ('{enemy_type}') can be personified and exaggerated. Tension resolution can be comedic inversion. Brand must be the straight-faced solution to the absurd problem.
    - Product Beauty: No human talent. The product is the sole subject. Every beat is a visual exploration of the object itself — texture, liquid movement, ingredient close-ups, material surfaces. Lighting and composition carry the emotional arc. VO is minimal and poetic. Hook is a visually arresting product macro shot.
    - Flatlay: No human talent. Product is shot from directly above on a clean, styled surface. Props (ingredients, botanicals, fabric swatches, tools) are arranged to reinforce brand codes. Color palette and negative space are primary expressive tools. Movement is subtle — a slow drift or single prop entering frame. Copy and text supers carry the informational load.
    - CGI/3D Product: No human talent. The product exists in a fully stylized, rendered environment. Physics can be non-literal (floating, slow-motion liquid, particle effects, material morphs). Brand world-building is the primary task — every surface, light source, and environmental element should reinforce brand identity. This is a brand film for the object, not a demo.

    INSTRUCTIONS:
    1. Acknowledge the input video_type ("{video_type}"). If it does not strictly match one of the defined types above, map it to the closest fit.
    2. Write an original, tailored 3-sentence conditioning directive for `video_type_conditioning_notes` that will guide subsequent ideation agents based on the rules for this format. DO NOT simply copy and paste the rules—translate them into actionable campaign instructions.
    3. Determine the 'format_physics_rules' object based on the rules applicable for this format.
    4. Provide detailed reasoning for this conditioning configuration under `reasoning`.
    5. List explicit directions or styles to NOT use under `suppressed_directions`.
    
    OUTPUT FORMAT:
    You must return STRICTLY JSON that matches this schema:
    {VideoTypeConditioningResult.model_json_schema()}
    """

    # 3. Call Gemini
    invoke_start = time.time()
    logger.info("Agent 5a: Preparing to call Gemini model=%s...", GEMINI_MODEL)
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info("Agent 5a: Gemini Client instantiated.")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": VideoTypeConditioningResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info("Agent 5a Gemini call completed | duration=%.2fs", api_duration)

        raw_json = response.text
        logger.info("Agent 5a: Raw JSON response received (len=%d). Cleaning...", len(raw_json))
        cleaned_json = _clean_json_string(raw_json)
        parsed_data = json.loads(cleaned_json)

        # Validate with Pydantic
        logger.info("Agent 5a: Validating JSON with Pydantic...")
        result = VideoTypeConditioningResult(**parsed_data)
        result.status = "completed"
        logger.info("Agent 5a: Pydantic validation successful.")

    except asyncio.TimeoutError as exc:
        api_duration = time.time() - invoke_start
        logger.error("Agent 5a: Call to Gemini TIMED OUT after %.2fs", api_duration)
        await _log_error(db, project_id, "video_type_conditioning_agent", "Gemini API Timeout")
        return VideoTypeConditioningResult(status="error", reason="Gemini API Timeout")
    except Exception as exc:
        logger.error("Error generating or parsing logic  |  error=%s", exc)
        await _log_error(db, project_id, "video_type_conditioning_agent", str(exc))
        return VideoTypeConditioningResult(status="error", reason=str(exc))

    total_duration = time.time() - start_time

    # 4. Save results
    try:
        # Save to Ideation
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "video_type_final": result.video_type_final,
                "video_type_conditioning_notes": result.video_type_conditioning_notes,
                "status.video_type_conditioning_agent": "completed",
                "updated_at": time.time()
            }},
            upsert=True
        )

        # Save to Pipeline Log
        pipeline_log = {
            "agent_key": "video_type_conditioning_agent",
            "status": "completed",
            "video_type_final": result.video_type_final,
            "reasoning": result.reasoning,
            "format_physics_rules": result.format_physics_rules.model_dump() if result.format_physics_rules else None,
            "suppressed_directions": result.suppressed_directions,
            "duration_secs": total_duration,
            "api_duration_secs": api_duration,
            "timestamp": time.time(),
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

    except Exception as e:
        logger.error("Failed to save results to DB | error=%s", e)
        return VideoTypeConditioningResult(status="error", reason="DB save failed")

    logger.info("Agent 5a completed successfully | duration=%.2fs", total_duration)
    return result

async def _log_error(db, project_id: str, agent_key: str, error_msg: str):
    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {f"status.{agent_key}": "error"}}
        )
        error_log = {
            "agent_key": agent_key,
            "status": "error",
            "error_msg": error_msg,
            "timestamp": time.time(),
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": error_log}},
            upsert=True
        )
    except Exception as e:
        logger.error("Failed to log error to DB | error=%s", e)
