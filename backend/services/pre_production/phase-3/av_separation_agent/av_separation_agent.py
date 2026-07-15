import json
import logging
import os
import time
from typing import List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

logger = logging.getLogger("zeroshot.phase3.av_separation_agent")


def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")


class AVSeparationDecision(BaseModel):
    window_id: str = Field(description="Window identifier, e.g. W01")
    visual_info: str = Field(description="what the visual communicates")
    vo_info: Optional[str] = Field(description="what VO communicates (null = silent window)")
    separation_principle: str = Field(description="e.g. 'visual carries sensory, VO carries emotional contrast'")
    tension_opportunity: bool = Field(description="Flag to indicate tension opportunity")


class AVSeparationResult(BaseModel):
    av_channel_map: List[AVSeparationDecision]
    reasoning: str
    status: str = "pending"


async def run_av_separation_agent(project_id: str, db, revision_prefix: str | None = None) -> AVSeparationResult:
    agent_key = "av_separation_agent"
    run_start = time.time()
    logger.info(f"Initializing Agent [4]... | project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error("Agent [4]: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"[{agent_key}] Fetching data for project_id={project_id}")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")

        script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
        if not script_doc:
            raise ValueError(f"Script document for '{project_id}' not found")

        phase_2_output = ideation_doc.get("phase_2_output", ideation_doc)
        approved_concepts = phase_2_output.get("approved_concepts", ideation_doc.get("approved_concepts", []))
        if not approved_concepts:
            raise ValueError("No approved concepts found in ideation document")

        concept_index = 0
        concept = approved_concepts[concept_index]
        concept_id = concept.get("concept_id", "Unknown Concept")

        format_group = phase_2_output.get("format_group", ideation_doc.get("format_group", "N"))
        is_visual = format_group == "V"
        beats = concept.get("composition_beats", []) if is_visual else concept.get("story_beats", [])
        beat_section_label = "COMPOSITION BEATS" if is_visual else "STORY BEATS"
        archetype = (concept.get("visual_motif", "") if is_visual else concept.get("archetype", ""))

        if is_visual:
            selected_motif = phase_2_output.get("selected_visual_motif", ideation_doc.get("selected_visual_motif", {}))
            micro_policy = selected_motif.get("visual_micro_policy", "")
        else:
            selected_archetype = phase_2_output.get("selected_archetype", ideation_doc.get("selected_archetype", {}))
            micro_policy = selected_archetype.get("micro_policy", "")

        master_timeline = script_doc.get("master_timeline", {})
        windows = master_timeline.get("windows", [])
        if not windows:
            raise ValueError("master_timeline.windows missing from script document")

        offer_constraints = script_doc.get("offer_constraints", {})
        vo_prohibition = offer_constraints.get("vo_prohibition", "")
        vo_language_directive = offer_constraints.get("vo_language_directive", "")
        visual_directive = offer_constraints.get("visual_directive", "")

        shot_list = script_doc.get("shot_list", [])
        if not shot_list:
            raise ValueError("shot_list missing from script document")

        logger.info(
            f"[{agent_key}] Extracted key inputs: concept_id={concept_id}, format_group={format_group}, "
            f"beats_count={len(beats)}, windows_count={len(windows)}, "
            f"shot_list_count={len(shot_list)}, archetype/motif='{archetype}'"
        )

        prompt_input = {
            "concept_id": concept_id,
            beat_section_label.lower().replace(" ", "_"): beats,
            "archetype": archetype,
            "micro_policy": micro_policy,
            "windows": windows,
            "shot_list": shot_list,
            "offer_constraints": {
                "vo_prohibition": vo_prohibition,
                "vo_language_directive": vo_language_directive,
                "visual_directive": visual_directive
            }
        }

        system_prompt = (
            "You are a channel architecture specialist for short-form video. Your job is to decide, for each time window, "
            "what information belongs in the visual channel and what belongs in the VO channel — enforcing the 1+1=3 principle. "
            "The two channels must never say the same thing. One track completes the other. Where the visual carries emotion, "
            "the VO carries function. Where the visual carries the product fact, the VO carries the human truth.\n\n"
            "You are grounded strictly in the shot list and story beats provided. You do not write VO lines. "
            "You determine what each channel communicates — the actual words and shots come from downstream agents."
        )

        user_prompt = f"""You are mapping channel separation for concept: {concept_id}

{beat_section_label}:
{json.dumps(beats, indent=2)}

{"VISUAL MOTIF" if is_visual else "ARCHETYPE"}:
{archetype}

{"VISUAL MOTIF MICRO-POLICY" if is_visual else "ARCHETYPE MICRO-POLICY"}:
{micro_policy}

MASTER TIMELINE:
{json.dumps(windows, indent=2)}

SHOT LIST (what is visually happening per window):
{json.dumps(shot_list, indent=2)}

OFFER CHANNEL CONSTRAINTS:
VO prohibition: {vo_prohibition}
VO language directive: {vo_language_directive}
Visual directive: {visual_directive}

TASK:
Step 1 — For each window, evaluate what information the visual is already communicating based on the shot list. State this explicitly as "visual carries: [X]".

Step 2 — Determine what complementary information the VO channel should carry for that window. The VO must not repeat or describe what the visual shows. It must supply a different dimension — if the visual is sensory, the VO is emotional; if the visual shows product application, the VO can carry the mythic meaning of the action. State this as "VO carries: [Y]".

Step 3 — For windows in the offer beat, apply the offer channel constraints without exception. If VO is prohibited, mark the window as silent in the VO channel regardless of what the visual communicates.

Step 4 — Identify windows where creative tension between channels is possible — where giving the copy an unexpected tone (irony, understatement, contrast) would make the combination more powerful than either channel alone. Flag these as "tension opportunity."

CONSTRAINTS (apply last):
- No window may have both channels communicating the same piece of information.
- Silent windows are valid and must be explicitly marked as silent, not left blank.
- Do not write actual VO lines or shot descriptions. Map information only.
- Output as a JSON array, one entry per window_id, with fields: window_id, visual_info, vo_info (null if silent), separation_principle, tension_opportunity (boolean).
"""

        prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
        if revision_prefix:
            prompt = f"{revision_prefix}\n\n{prompt}"

        invoke_start = time.time()
        logger.info(f"Agent [4]: Preparing to call Gemini model={GEMINI_MODEL}...")
        try:
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
            )
            logger.info(f"Agent [4]: Gemini Client instantiated. Sending prompt...")

            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": AVSeparationResult.model_json_schema(),
                    "automatic_function_calling": {"disable": True},
                }
            )

            api_duration = time.time() - invoke_start
            logger.info(f"Agent [4]: Gemini API call completed in {api_duration:.2f}s")
            logger.debug(f"Agent [4]: Raw response length={len(response.text)} chars")

            cleaned_json = _clean_json_string(response.text)
            parsed_data = json.loads(cleaned_json)
            logger.info(f"Agent [4]: Successfully parsed JSON response.")

            result = AVSeparationResult(**parsed_data)
            result.status = "completed"
            logger.info(f"Agent [4]: Successfully validated structured output with Pydantic.")

        except Exception as llm_error:
            if "503" in str(llm_error) or "429" in str(llm_error):
                logger.error(f"Gemini API overloaded: {llm_error}")
            else:
                logger.error(f"Agent [4]: Gemini invocation failed: {llm_error}", exc_info=True)
            raise

        logger.info("Agent [4]: Updating SCRIPT and PIPELINE collections...")
        await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {
                "$set": {
                    "project_id": str(project_id),
                    "av_channel_map": [item.model_dump() for item in result.av_channel_map],
                }
            },
            upsert=True
        )

        pipeline_log = {
            "agent_id": agent_key,
            "timestamp": time.time(),
            "status": "completed",
            "execution_time_seconds": round(time.time() - run_start, 2),
            "reasoning": result.reasoning
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        logger.info(f"Agent [4]: Successfully persisted results for project_id={project_id}")
        return result

    except Exception as e:
        if "503" in str(e) or "429" in str(e):
            logger.error(f"Execution failed due to API bottleneck: {e}")
        else:
            logger.error(f"Agent [4]: Execution failed: {e}", exc_info=True)
        pipeline_log = {
            "agent_id": agent_key,
            "timestamp": time.time(),
            "status": "failed",
            "error": str(e),
            "execution_time_seconds": round(time.time() - run_start, 2)
        }
        try:
            await db[PIPELINE_COLLECTION].update_one(
                {"project_id": str(project_id)},
                {"$push": {"agent_logs": pipeline_log}},
                upsert=True
            )
        except Exception as db_err:
            logger.error(f"Agent [4]: Failed to log error to PIPELINE_COLLECTION: {db_err}")
        raise
