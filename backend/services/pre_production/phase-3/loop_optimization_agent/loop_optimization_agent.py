import os
import json
import time
import logging
from typing import List, Optional, Any, Dict
from pydantic import BaseModel
from bson import ObjectId
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

# Global Configuration Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")

logger = logging.getLogger("zeroshot.phase3.loop_optimization_agent")


class LoopRevisionRequest(BaseModel):
    dimension: str
    target_agent: str
    instruction: str


class LoopOptimizationResponse(BaseModel):
    visual_continuity_pass: bool
    tonal_continuity_pass: bool
    audio_continuity_pass: bool
    curiosity_loop_pass: bool
    revision_requests: Optional[List[LoopRevisionRequest]] = None
    status: str = "completed"


def _clean_json_string(s: str) -> str:
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


async def run_loop_optimization_agent(project_id: str, db, revision_prefix: str | None = None) -> LoopOptimizationResponse:
    logger.info(f"Initializing Agent 9 [Loop Optimization Agent] for project_id={project_id}...")

    # Data Fetching
    logger.info(f"Fetching data from IDEATION_COLLECTION and SCRIPT_COLLECTION for project_id={project_id}...")
    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
    if not ideation_doc:
        raise ValueError(f"No ideation document found for project_id={project_id}")

    script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
    if not script_doc:
        raise ValueError(f"No script document found for project_id={project_id}")

    # Extract required inputs safely
    phase_2_output = ideation_doc.get("phase_2_output", {})
    approved_concepts = phase_2_output.get("approved_concepts", ideation_doc.get("approved_concepts", []))
    concept = approved_concepts[0] if approved_concepts else {}
    concept_id = concept.get("concept_id", "Unknown")
    archetype_name = concept.get("archetype", "")
    
    master_timeline = script_doc.get("master_timeline", {})
    windows = master_timeline.get("windows", []) if isinstance(master_timeline, dict) else []
    
    shot_list = script_doc.get("shot_list", [])
    vo_script = script_doc.get("vo_script", [])
    
    audio_design = script_doc.get("audio_design", {})
    audio_windows = audio_design.get("windows", []) if isinstance(audio_design, dict) else []
    music_mood_curve = audio_design.get("music_mood_curve", "") if isinstance(audio_design, dict) else ""
    
    pacing_map = script_doc.get("pacing_map", {})
    pacing_windows = pacing_map.get("windows", []) if isinstance(pacing_map, dict) else []
    energy_curve = pacing_map.get("energy_curve", "") if isinstance(pacing_map, dict) else ""

    logger.info(
        f"Agent 9 [Loop Optimization Agent]: Extracted key inputs. "
        f"Concept={concept_id}, Archetype='{archetype_name}', "
        f"Windows={len(windows)}, Shots={len(shot_list)}, VOSegs={len(vo_script)}, "
        f"AudioWindows={len(audio_windows)}, PacingWindows={len(pacing_windows)}"
    )

    prompt = f"""You are a loop architect for Instagram Reels. Your job is to evaluate whether the assembled script creates a seamless, rewarding loop — where the last moment of the video flows organically back into the first. Loop design is structural, not cosmetic. You evaluate three dimensions: visual continuity between last and first frame, tonal continuity between end and opening emotional energy, and audio continuity between closing and opening soundscape.

When you detect a loop failure, you issue a targeted revision request to the responsible upstream agent. You do not rewrite anything yourself. You also evaluate whether the opening hook contains enough unresolved tension to make a second viewing feel rewarding — curiosity loop design.

You are grounded strictly in the assembled script provided.

You are evaluating loop continuity for concept: {concept_id}

CONCEPT ARCHETYPE:
{archetype_name}

MASTER TIMELINE (opening and closing windows are primary focus):
{json.dumps(windows, indent=2)}

SHOT LIST:
{json.dumps(shot_list, indent=2)}

VO SCRIPT:
{json.dumps(vo_script, indent=2)}

AUDIO DESIGN:
Audio windows: {json.dumps(audio_windows, indent=2)}
Music mood curve: {music_mood_curve}

PACING MAP (energy levels at start and end):
{json.dumps(pacing_windows, indent=2)}
Energy curve: {energy_curve}

TASK:
Step 1 — Visual continuity: Compare the last shot in the shot_list against the first shot. Do they share a compositional element — a subject position, a framing, a colour temperature, a physical object — that would make the transition feel organic on loop? If not, issue a revision request to visual_sequencing_agent specifying exactly what compositional echo to introduce in the closing shot.

Step 2 — Tonal continuity: Compare the emotional energy of the final window against the opening window using the pacing_map. For the Ritual archetype, the Offering beat (warm, inviting) looping back to the Intrigue opening (aspirational, curious) is tonally compatible. A gap larger than 2 energy points creates emotional whiplash. If the gap is too wide, issue a revision request to voiceover_writer_agent to introduce a subtle callback phrase in the closing VO line that echoes the opening hook register.

Step 3 — Audio continuity: Compare the closing audio design (final window ambient and music) against the opening audio design (first window). Does the music fade or transition in a way that flows back into the opening without a harsh cut? If not, issue a revision request to audio_design_agent requesting a crossfade-compatible ending — specifically what texture the final 1-2 seconds should fade to.

Step 4 — Curiosity loop evaluation: Does the opening hook contain unresolved tension — a question not yet answered, a visual state not yet explained — that makes a second viewing feel rewarding rather than redundant? If the hook resolves too completely on first watch, flag it for compression or reframing and issue a note to voiceover_writer_agent.

CONSTRAINTS (apply last):
- Evaluate only the first and last 2-3 seconds of the assembled script for loop dimensions. Do not review the full middle section.
- Every revision request must name the specific target agent and provide an actionable instruction, not a vague direction.
- If all three loop dimensions pass, output null for revision_requests.
- Output as a JSON object with fields: visual_continuity_pass (boolean), tonal_continuity_pass (boolean), audio_continuity_pass (boolean), curiosity_loop_pass (boolean), revision_requests (array or null)."""

    if revision_prefix:
        prompt = f"{revision_prefix}\n\n{prompt}"

    invoke_start = time.time()
    logger.info(f"Agent 9: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 9: Gemini Client instantiated. Sending prompt...")
  
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": LoopOptimizationResponse.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
  
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 9: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 9: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 9: Successfully parsed JSON response.")

        result = LoopOptimizationResponse(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 9: Successfully validated structured output with Pydantic.")

        # Database Updates
        logger.info(f"Agent 9: Updating SCRIPT and PIPELINE collections for project_id={project_id}...")
        
        # Write final output to SCRIPT_COLLECTION
        # loop_revision_requests is written as a top-level field so the orchestrator can read it directly
        loop_revision_requests_payload = (
            [r.model_dump() for r in result.revision_requests]
            if result.revision_requests else []
        )
        await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "loop_optimization": result.model_dump(),
                "loop_revision_requests": loop_revision_requests_payload,
            }},
            upsert=True
        )

        # Push to PIPELINE_COLLECTION using standard pipeline_log object
        pipeline_log = {
            "agent": "loop_optimization_agent",
            "status": getattr(result, 'status', 'completed'),
            "duration": api_duration,
            "timestamp": time.time(),
            "reasoning": "Completed loop continuity evaluation across visual, tonal, and audio dimensions."
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )
        logger.info(f"Agent 9: Updates completed successfully.")

        total_duration = time.time() - invoke_start
        logger.info(f"Agent 9: Total run duration: {total_duration:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Agent 9: Execution failed with error: {str(e)}")
        raise
