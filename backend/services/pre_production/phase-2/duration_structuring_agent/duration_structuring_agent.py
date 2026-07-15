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
logger = logging.getLogger("zeroshot.duration_structuring")

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")

# ── Pydantic Models for Structured Output ────────────────────────────────────

class NarrativeBudget(BaseModel):
    total_seconds: int = Field(description="Total video length in seconds")
    hook_seconds: float = Field(description="Seconds allocated to the hook")
    tension_seconds: float = Field(description="Seconds allocated to tension development")
    demo_seconds: float = Field(description="Seconds allocated to product demonstration")
    payoff_seconds: float = Field(description="Seconds allocated to emotional payoff")
    offer_seconds: float = Field(description="Seconds allocated to offer communication")
    compression_flags: List[str] = Field(description="Functions that cannot be adequately served at this duration")

class DurationStructuringResult(BaseModel):
    status: Literal["completed", "error", "skipped"]
    reason: Optional[str] = None
    
    # Output Schema
    narrative_budget: Optional[NarrativeBudget] = None
    
    # Pipeline Log Fields
    reasoning: Optional[str] = Field(None, description="Reasoning behind budget allocation")
    beat_compression_analysis: Optional[str] = Field(None, description="Analysis of compressed or cut functions")
    pacing_notes: Optional[str] = Field(None, description="Strategic pacing notes for execution")

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

async def run_duration_structuring_agent(project_id: str, db) -> DurationStructuringResult:
    """
    Agent 6: Duration Structuring.
    RUN CONDITION: ALWAYS.
    Converts video_length_seconds into a precise narrative resource budget.
    """
    logger.info("Initializing Agent 6 (Duration Structuring) | project_id=%s", project_id)
    start_time = time.time()

    # 1. Fetch relevant data
    logger.info(f"Agent 6: Fetching data for project_id={project_id}")
    try:
        proj_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not proj_doc:
            logger.error(f"Agent 6: Project not found: {project_id}")
            raise ValueError(f"Project not found: {project_id}")
            
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}
        logger.info(f"Agent 6: DB fetch successful. Found project and ideation docs.")
    except Exception as e:
        logger.error("Agent 6: Failed to fetch project doc | error=%s", e)
        return DurationStructuringResult(status="error", reason=str(e))

    # Retrieve inputs matching schema
    video_length_seconds = proj_doc.get("video_length_seconds")
    number_of_shots = proj_doc.get("number_of_shots")
    video_type_final = ideation_doc.get("video_type_final", "Unknown")
    
    constraint_graph = ideation_doc.get("constraint_graph", {})
    feasibility_envelope = constraint_graph.get("feasibility_envelope", "")
    priority_directives = ideation_doc.get("priority_directives", {})

    logger.info(f"Agent 6: Extracted inputs | video_length_seconds={video_length_seconds}, video_type_final={video_type_final}")

    if not video_length_seconds:
        logger.warning(f"Agent 6: Skipping - video_length_seconds is not available | project_id={project_id}")
        return DurationStructuringResult(status="skipped", reason="video_length_seconds is missing")

    if number_of_shots is None:
        logger.warning(f"Agent 6: number_of_shots not found in project doc | project_id={project_id}")

    # 2. Construct prompt
    prompt = f"""
    You are the Duration Structuring Agent (Agent 6) for an advertising campaign.
    
    PURPOSE:
    Convert the provided video length ({video_length_seconds} seconds) into a precise second-by-second narrative resource budget 
    across five functions: hook, tension, demo, payoff, offer. Every second is a strategic asset. Flag functions 
    that cannot be adequately served at this duration and specify what must be compressed or cut.
    
    INPUT DATA:
    Project Context:
      video_length_seconds: {video_length_seconds}
      number_of_shots: {number_of_shots}
      shot_duration_seconds: 8  (fixed — each shot is exactly 8 seconds)
      video_type_final: "{video_type_final}"
      constraint_graph.feasibility_envelope: "{feasibility_envelope}"
      priority_directives: {json.dumps(priority_directives)}

    SHOT ATOMICITY CONSTRAINT (highest priority — apply before all allocation logic):
    - Each shot is exactly 8 seconds and cannot be subdivided.
    - ALL five budget allocations MUST be multiples of 8 seconds.
      Valid values: 8, 16, 24, 32, 40, … (integer multiples of 8 only).
      Allocating 10 or 6 seconds to any function is INVALID.
    - The sum hook_seconds + tension_seconds + demo_seconds + payoff_seconds + offer_seconds
      must equal exactly {video_length_seconds} seconds ({number_of_shots} shots × 8 seconds).
    - If a function's natural allocation is not divisible by 8, round to the nearest multiple
      of 8 and compensate from an adjacent function. Document the adjustment in compression_flags.

    NARRATIVE FUNCTIONS ALLOCATION LOGIC:
    1. Hook establishment: The window to stop the scroll and establish the conflict or curiosity. For feeds/reels: complete within first 20%. In-stream: can extend to 25%. Minimum viable hook: 2 seconds. Never exceed 35%.
    2. Tension development: The body. Short formats (under 15s): compress or eliminate. Over 20s: primary investment window.
    3. Product demonstration: The moment the product appears. Must be specific and physical. Always required. Minimum: 2 seconds. Include preferred scene by default.
    4. Emotional payoff: Resolution beat. Can be joyful, defiant, calm, etc depending on archetype. Never cut this. Minimum: 2 seconds.
    5. Offer communication: Commercial mechanic. Should feel like a consequence of the emotional payoff. Maximum: 20% of duration.

    INSTRUCTIONS:
    - Allocate seconds to the five functions above. The sum MUST strictly equal {video_length_seconds} seconds ({number_of_shots} × 8). Every individual allocation must be a multiple of 8 seconds (no fractional shots).
    - If any function is compressed or cut due to length, list it in `compression_flags`.
    - Provide a concise explanation under `reasoning`.
    - Provide a `beat_compression_analysis` describing the implications of short duration on specific functions.
    - Provide `pacing_notes` covering rhythm and execution style for this duration and video type.

    OUTPUT FORMAT:
    Output STRICTLY JSON matching this schema. NO markdown formatting.
    {DurationStructuringResult.model_json_schema()}
    """

    # 3. Call Gemini
    invoke_start = time.time()
    logger.info(f"Agent 6: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 6: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": DurationStructuringResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 6: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 6: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 6: Successfully parsed JSON response.")

        result = DurationStructuringResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 6: Successfully validated structured output with Pydantic.")

    except asyncio.TimeoutError:
        logger.error("Agent 6: Call to Gemini API timed out")
        return DurationStructuringResult(status="error", reason="Gemini API Timeout")
    except Exception as exc:
        logger.error("Agent 6: Error generating/parsing | error=%s", exc)
        return DurationStructuringResult(status="error", reason=str(exc))

    total_duration = time.time() - start_time

    # 4. Save results
    logger.info(f"Agent 6: Updating IDEATION and PIPELINE collections for project_id={project_id}...")
    try:
        if result.narrative_budget:
            await db[IDEATION_COLLECTION].update_one(
                {"project_id": str(project_id)},
                {"$set": {
                    "narrative_budget": result.narrative_budget.model_dump(),
                    "status.duration_structuring_agent": "completed",
                    "updated_at": time.time()
                }},
                upsert=True
            )
            logger.info("Agent 6: IDEATION collection updated with narrative_budget.")

        pipeline_log = {
            "agent_key": "duration_structuring_agent",
            "status": "completed",
            "reasoning": result.reasoning,
            "beat_compression_analysis": result.beat_compression_analysis,
            "pacing_notes": result.pacing_notes,
            "duration_secs": total_duration,
            "api_duration_secs": api_duration,
            "timestamp": time.time(),
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )
        logger.info("Agent 6: PIPELINE collection updated with execution log.")

    except Exception as e:
        logger.error(f"Agent 6: Failed to save results to DB | error={e}")
        return DurationStructuringResult(status="error", reason="DB save failed")

    logger.info(f"Agent 6 completed successfully | total_duration={total_duration:.2f}s | project_id={project_id}")
    return result
