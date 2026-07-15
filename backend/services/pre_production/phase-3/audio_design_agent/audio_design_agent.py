import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
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

logger = logging.getLogger("zeroshot.phase3.audio_design_agent")


class AudioWindow(BaseModel):
    window_id: str
    ambient: str
    sfx: str
    music_directive: str
    silence: bool
    redundancy_verified: bool


class AudioDesignResponse(BaseModel):
    windows: List[AudioWindow]
    music_mood_curve: str
    sound_off_compliant: bool
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


async def run_audio_design_agent(project_id: str, db, revision_prefix: str | None = None) -> AudioDesignResponse:
    logger.info(f"Initializing Agent 7 [Audio Design Agent] for project_id={project_id}...")

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
    approved_concepts = phase_2_output.get("approved_concepts", [])
    concept = approved_concepts[0] if approved_concepts else {}
    concept_id = concept.get("concept_id", "Unknown")
    
    format_group = phase_2_output.get("format_group", ideation_doc.get("format_group", "N"))
    is_visual = format_group == "V"
    if is_visual:
        selected_motif = phase_2_output.get("selected_visual_motif", ideation_doc.get("selected_visual_motif", {}))
        archetype_name = selected_motif.get("selected_motif", concept.get("visual_motif", ""))
        micro_policy = selected_motif.get("visual_micro_policy", "")
        failure_modes = selected_motif.get("failure_modes", [])
    else:
        selected_archetype = phase_2_output.get("selected_archetype", ideation_doc.get("selected_archetype", {}))
        archetype_name = concept.get("archetype", "")
        micro_policy = selected_archetype.get("micro_policy", "")
        failure_modes = selected_archetype.get("failure_modes", [])

    platform_rules = phase_2_output.get("platform_rules", ideation_doc.get("platform_rules", {}))
    soft_rules = platform_rules.get("soft_rules", [])
    authenticity_signal = platform_rules.get("authenticity_signal", "")

    master_timeline = script_doc.get("master_timeline", {})
    windows = master_timeline.get("windows", []) if isinstance(master_timeline, dict) else []

    shot_list = script_doc.get("shot_list", [])
    vo_script = script_doc.get("vo_script", [])
    av_channel_map = script_doc.get("av_channel_map", [])

    logger.info(
        f"Agent 7 [Audio Design Agent]: Extracted key inputs. "
        f"Concept={concept_id}, Archetype='{archetype_name}', "
        f"Windows={len(windows)}, Shots={len(shot_list)}, VO Segments={len(vo_script)}"
    )

    # Prompt Setup (Unified as per user guide)
    prompt = f"""You are a non-verbal audio director for short-form video. You design everything the viewer hears that is not spoken words: ambient texture, sound effects, music mood, and deliberate silence. You work across four sub-layers simultaneously and you sequence them against the visual and VO channels to create coherence, not collision.

Your most important constraint is sound-off redundancy: any audio element that carries narrative meaning must be backed up by a visual or text super that communicates the same meaning independently. Atmospheric audio needs no backup — it enhances but is not essential. You are grounded strictly in the provided shot list, VO script, archetype, and platform requirements.

You are designing the audio layer for concept: {concept_id}

ARCHETYPE:
{archetype_name}

ARCHETYPE MICRO-POLICY (governs audio tone and pacing):
{micro_policy}
Failure modes to avoid: {json.dumps(failure_modes)}

PLATFORM REQUIREMENTS:
Soft rules: {json.dumps(soft_rules)}
Authenticity signal: {authenticity_signal}

MASTER TIMELINE:
{json.dumps(windows, indent=2)}

SHOT LIST (what is visually happening per window):
{json.dumps(shot_list, indent=2)}

VO SCRIPT (spoken words per window):
{json.dumps(vo_script, indent=2)}

AV CHANNEL MAP (authoritative VO assignments per window — silence must not conflict with these):
{json.dumps(av_channel_map, indent=2)}

TASK:
Step 1 — Sub-layer 1 (Ambient texture): For each window, assign an ambient sound environment that reinforces the visual context without competing with it. Corporate/sterile windows get AC hum, keyboard clicks, or fluorescent flicker. Personal ritual windows get near-silence, soft breath, or faint botanical texture.

Step 2 — Sub-layer 2 (Sound effects): Assign SFX to every physical product interaction in the shot list. For a Ritual archetype, jar opening and direct application textures must be ASMR-grade — slow, tactile, precisely rendered. For Concept E specifically, the Slack ping is a narrative device, not atmospheric audio — it carries story meaning and requires visual and text super redundancy. Flag it.

Step 3 — Sub-layer 3 (Music mood curve): Write one music directive for the full 32 seconds as a mood and energy instruction — not a track name. Specify: BPM range, instrumentation texture, key energy inflection points (e.g. drops to silence at 14s, warm pads re-enter at 18s), and how the music responds to the VO density.

Step 4 — Sub-layer 4 (Silence design): Identify windows where deliberate silence is the correct choice — particularly the Symbolic Transformation window, where the Ritual archetype demands an unbroken trance-like aesthetic. Silence is a design choice, not an absence. CRITICAL: You must NOT set silence=true for any window where the AV channel map assigns a non-null VO line. Silence here means music/SFX silence, not VO silence — VO is controlled by the AV channel map, not by you.

Step 5 — Sound-off redundancy check: For every SFX that carries narrative meaning (not atmospheric), verify the shot_list and vo_script for the same window provide redundant communication of the same meaning. If they do not, flag for Agent 3 revision.

Step 6 — Collision prevention: Where the VO script has a spoken line in a window, the music directive for that window must recede. Inverse relationship between VO density and audio intensity is mandatory.

CONSTRAINTS (apply last):
- Do not specify actual track names or licensed music — mood directives only.
- Sound-off compliance: the script must work for viewers who hear nothing. Verify this before outputting.
- Output as a JSON object with fields: windows (array, one per window_id with: ambient, sfx, music_directive, silence boolean, redundancy_verified boolean), music_mood_curve (string), sound_off_compliant (boolean).
"""

    # Model Invocation (STRICT SYNTAX REQUIRED)
    invoke_start = time.time()
    logger.info(f"Agent 7: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 7: Gemini Client instantiated. Sending prompt...")
        
        if revision_prefix:
            existing_script_doc = await db[SCRIPT_COLLECTION].find_one(
                {"project_id": str(project_id)}, {"audio_design": 1}
            )
            previous_audio = (existing_script_doc or {}).get("audio_design") or {}
            if previous_audio:
                revision_block = (
                    revision_prefix
                    + "\n\nYOUR PREVIOUS AUDIO DESIGN OUTPUT — copy every window VERBATIM except the windows listed above:\n"
                    + json.dumps(previous_audio, indent=2)
                )
            else:
                revision_block = revision_prefix
            prompt = revision_block + "\n\n" + prompt

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": AudioDesignResponse.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
  
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 7: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 7: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 7: Successfully parsed JSON response.")

        result = AudioDesignResponse(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 7: Successfully validated structured output with Pydantic.")

    except Exception as e:
        logger.error(f"Agent 7 Gemini API call failed: {e}")
        raise RuntimeError(f"Agent 7 Gemini API call failed: {e}")

    # Database Update
    logger.info("Updating SCRIPT and PIPELINE collections...")
    
    script_update_result = await db[SCRIPT_COLLECTION].update_one(
        {"project_id": str(project_id)},
        {"$set": {"audio_design": result.model_dump()}}
    )

    if script_update_result.modified_count == 0 and script_update_result.matched_count == 0:
        logger.error(f"Failed to update {SCRIPT_COLLECTION} for project_id={project_id}. Document not found.")
        raise ValueError(f"Failed to update script document for project_id={project_id}")

    pipeline_log = {
        "agent": "audio_design_agent",
        "timestamp": time.time(),
        "duration": api_duration,
        "status": "completed"
    }
    
    await db[PIPELINE_COLLECTION].update_one(
        {"project_id": str(project_id)},
        {"$push": {"agent_logs": pipeline_log}},
        upsert=True
    )
    
    logger.info(f"Agent 7 [Audio Design Agent] completed successfully for project_id={project_id} in {api_duration:.2f}s")
    
    return result
