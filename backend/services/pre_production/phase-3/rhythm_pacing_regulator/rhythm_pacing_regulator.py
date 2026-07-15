import os
import sys
import json
import time
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# ----------------------------------------------------------------------
# 1. Global Setup & Configuration (STRICT)
# ----------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "zeroshot")

logger = logging.getLogger("zeroshot.phase3.rhythm_pacing_regulator")
logger.setLevel(logging.INFO)
# fallback logging config if root isn't configured
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# ----------------------------------------------------------------------
# 2. Pydantic Models for output schema
# ----------------------------------------------------------------------

# Submodels for Script Pacing Map
class PacingWindowResult(BaseModel):
    window_id: str
    cut_density: str
    vo_density: str
    micro_pause_before: Optional[float] = None
    energy_level: int
    pacing_note: Optional[str] = None

class RevisionRequestLogFields(BaseModel):
    dimension: str
    target_agent: str
    window_id: str
    instruction: str

class RhythmPacingMap(BaseModel):
    windows: List[PacingWindowResult]
    energy_curve: str
    revision_requests: Optional[List[RevisionRequestLogFields]] = None

# Submodels for Pipeline Logs
class CutDensityAuditItem(BaseModel):
    window_id: str
    cuts: int
    verdict: str

class VODensityAuditItem(BaseModel):
    window_id: str
    word_count: int
    verdict: str

class PipelineRevisionLog(BaseModel):
    dimension: str
    target_agent: str
    rationale: str

class PipelineLogFields(BaseModel):
    reasoning: str
    energy_curve_analysis: str
    cut_density_audit: List[CutDensityAuditItem]
    vo_density_audit: List[VODensityAuditItem]
    revision_requests_log: List[PipelineRevisionLog]

# Unified Result Model
class RhythmPacingResultModel(BaseModel):
    pacing_map: RhythmPacingMap
    pipeline_log: PipelineLogFields
    status: Optional[str] = None


def _clean_json_string(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

# ----------------------------------------------------------------------
# 3. Prompt Template
# ----------------------------------------------------------------------
PROMPT_TEMPLATE = """
You are a rhythm and pacing regulator for short-form video. You are the first agent to evaluate the assembled script as a complete rhythmic whole — not channel by channel, but as a unified sensory experience. You regulate four dimensions simultaneously: cut frequency, VO density, micro-pause placement, and the emotional energy curve across the full timeline.

Your threshold is calibrated to "optimize for speed, not perfection" — you target rhythm failures that would cause viewer drop-off, not cosmetic imperfections. You issue revision requests to specific upstream agents with specific instructions. You do not rewrite content yourself. You diagnose and assign.

You are grounded strictly in the assembled script channels provided.

You are regulating rhythm and pacing for concept: {concept_id}

CONCEPT CATEGORY:
{concept_category}

CONCEPT ARCHETYPE:
{concept_archetype}

ARCHETYPE MICRO-POLICY (pacing standards for this archetype):
{micro_policy}

PLATFORM HOOK RULE:
{hook_window_rule}
Platform soft rules: {soft_rules}

ASSEMBLED SCRIPT (all channels):
Master timeline: {master_timeline}
Shot list (cut structure): {shot_list}
AV channel map (VO density): {av_channel_map}
VO script (word counts): {vo_script}
Audio design (music energy): {audio_design_windows}
Music mood curve: {audio_design_music_curve}

TASK:
Step 1 — Cut density audit: For each window in the shot_list, identify how many cuts or camera setups occur. Apply category-specific standards:
- PITCH: higher cut density acceptable in hook window (W01), must slow to maximum 1 cut per 3s in payoff.
- PLAY: moderate cut density throughout, must not break the social moment in the offer window.
- PLUNGE: maximum 1 cut per 4s throughout except where structurally essential.
For a Ritual archetype: more than 2 cuts in any 3-second window is a violation regardless of category. Flag violations with a revision request to visual_sequencing_agent.

Step 2 — VO density audit: For each window in the vo_script, calculate words-per-second. Natural speaking pace is approximately 2.5-3 words per second. A window with more than 3.5 words per second is too dense for a Ritual archetype — it breaks the trance. A window marked silent in the av_channel_map but carrying VO in the vo_script is a channel conflict error. Flag both with revision requests to voiceover_writer_agent.

Step 3 — Micro-pause placement: Identify 2-3 moments in the timeline where deliberate silence before a VO line would heighten attention. The canonical position is immediately before the Symbolic Transformation window. Write micro_pause_before values (in seconds) for each identified window.

Step 4 — Energy curve: Score each window 1-5 for emotional energy using: shot complexity, VO density, SFX presence, and beat function. Plot the arc. A well-formed Ritual arc: opens at 2-3, peaks at 4-5 during the Symbolic Transformation window (14-18s), resolves to 2-3 at the Offering beat. A flat curve (all windows within 1 point of each other) is a pacing failure. Flag flat curves with revision requests to both visual_sequencing_agent and audio_design_agent.

CONSTRAINTS (apply last):
- You do not rewrite content. You issue revision requests only.
- Every revision request must specify: dimension (cut/vo/pause/energy), target_agent, window_id, and a specific actionable instruction.
- Output exactly to the required JSON schema, populated with pacing_map data and the pipeline_log audit fields.
"""

# ----------------------------------------------------------------------
# 4. Main Runner Logic
# ----------------------------------------------------------------------
async def run_rhythm_pacing_regulator_agent(project_id: str, db, revision_prefix: str | None = None) -> Dict[str, Any]:
    logger.info(f"Initializing Agent 8 [rhythm_pacing_regulator] for project_id={project_id}...")
    
    start_time = time.time()
    
    try:
        # 1. Fetch DB Docs
        logger.info(f"Fetching data for project_id={project_id} from {IDEATION_COLLECTION} and {SCRIPT_COLLECTION}...")
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
        
        if not ideation_doc:
            raise ValueError(f"No ideation document found for project_id={project_id}")
        if not script_doc:
            raise ValueError(f"No script document found for project_id={project_id}")
            
        phase2_out = ideation_doc.get("phase_2_output", {})
        concepts = phase2_out.get("approved_concepts", [])
        concept = concepts[0] if concepts else {}
        
        format_group = phase2_out.get("format_group", ideation_doc.get("format_group", "N"))
        is_visual = format_group == "V"
        if is_visual:
            selected_motif = phase2_out.get("selected_visual_motif", ideation_doc.get("selected_visual_motif", {}))
            _archetype_micro_policy = selected_motif.get("visual_micro_policy", "None")
        else:
            selected_archetype = phase2_out.get("selected_archetype", {})
            _archetype_micro_policy = selected_archetype.get("micro_policy", "None")
        platform_rules = phase2_out.get("platform_rules", {})
        
        # Script Channels
        master_timeline = script_doc.get("master_timeline", {}).get("windows", [])
        shot_list = script_doc.get("shot_list", {})
        av_channel_map = script_doc.get("av_channel_map", {})
        vo_script = script_doc.get("vo_script", {})
        audio_design = script_doc.get("audio_design", {})
        
        # Extracted logging strings using strictly single quotes inside formatting as per constraints
        concept_id_str = concept.get('concept_id', 'unknown_concept')
        concept_cat_str = concept.get('category', 'unknown_category')
        
        logger.info(f"Agent [8]: Extracted Concept ID: {concept_id_str}, Category: {concept_cat_str}")
        logger.info(f"Agent [8]: Retrieved {len(master_timeline)} timeline windows.")
        
        prompt = PROMPT_TEMPLATE.format(
            concept_id=concept_id_str,
            concept_category=concept_cat_str,
            concept_archetype=concept.get("archetype", "unknown_archetype"),
            micro_policy=_archetype_micro_policy,
            hook_window_rule=platform_rules.get("hook_window_rule", "None"),
            soft_rules=json.dumps(platform_rules.get("soft_rules", [])),
            master_timeline=json.dumps(master_timeline),
            shot_list=json.dumps(shot_list),
            av_channel_map=json.dumps(av_channel_map),
            vo_script=json.dumps(vo_script),
            audio_design_windows=json.dumps(audio_design.get("windows", [])),
            audio_design_music_curve=audio_design.get("music_mood_curve", "None")
        )
        
        if revision_prefix:
            prompt = f"{revision_prefix}\n\n{prompt}"
            
        # 2. Call Gemini
        invoke_start = time.time()
        logger.info(f"Agent [8]: Preparing to call Gemini model={GEMINI_MODEL}...")
        try:
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
            )
            logger.info("Agent [8]: Gemini Client instantiated. Sending prompt...")
      
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": RhythmPacingResultModel.model_json_schema(),
                    "automatic_function_calling": {"disable": True},
                }
            )
            
            api_duration = time.time() - invoke_start
            logger.info(f"Agent [8]: Gemini API call completed in {api_duration:.2f}s")
            logger.debug(f"Agent [8]: Raw response length={len(response.text)} chars")

            cleaned_json = _clean_json_string(response.text)
            parsed_data = json.loads(cleaned_json)
            logger.info("Agent [8]: Successfully parsed JSON response.")

            result = RhythmPacingResultModel(**parsed_data)
            result.status = "completed"
            logger.info("Agent [8]: Successfully validated structured output with Pydantic.")
            
        except Exception as api_err:
            logger.error(f"Agent [8] Gemini API failure: {api_err}")
            raise RuntimeError(f"Failed to generate pacing map: {api_err}")

        # 3. Update Datastore
        logger.info(f"Updating {SCRIPT_COLLECTION} and {PIPELINE_COLLECTION} collections...")
        pacing_map_payload = result.pacing_map.model_dump()
        pipeline_log_payload = result.pipeline_log.model_dump()
        
        # Add metadata to pipeline log payload
        pipeline_log_payload["agent_name"] = "rhythm_pacing_regulator"
        pipeline_log_payload["execution_duration_sec"] = api_duration
        pipeline_log_payload["timestamp"] = time.time()

        # Update SCRIPT uniquely via $set (not nested under explicit "script." key)
        await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {"pacing_map": pacing_map_payload}},
            upsert=True
        )
        
        # Update PIPELINE uniquely via $push into agent_logs
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log_payload}},
            upsert=True
        )
        logger.info("Agent [8]: DB updates successful.")

        total_duration = time.time() - start_time
        logger.info(f"Agent [8] execution completed entirely in {total_duration:.2f}s")

        return {
            "status": "success",
            "pacing_map": pacing_map_payload,
            "pipeline_log": pipeline_log_payload
        }
        
    except Exception as e:
        logger.error(f"Agent [8] Error during execution: {e}")
        raise e
