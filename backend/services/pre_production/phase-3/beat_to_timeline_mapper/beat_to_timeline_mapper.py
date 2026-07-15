import json
import logging
import os
import time
from typing import List

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

logger = logging.getLogger("zeroshot.phase3.beat_to_timeline_mapper")


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


class TimelineWindow(BaseModel):
    id: str = Field(description="Window identifier, e.g. W01")
    start_s: float = Field(description="Window start in seconds")
    end_s: float = Field(description="Window end in seconds")
    beat_label: str = Field(description="Narrative beat label")
    beat_function: str = Field(description="Narrative function for this window")


class MasterTimeline(BaseModel):
    windows: List[TimelineWindow]


class BeatExpansionLogItem(BaseModel):
    beat: str
    start_s: float
    end_s: float
    rationale: str


class BeatToTimelineMapperResult(BaseModel):
    master_timeline: MasterTimeline
    reasoning: str
    beat_expansion_log: List[BeatExpansionLogItem]
    rationale: str
    timing_decisions: str
    status: str = "pending"


async def run_beat_to_timeline_mapper_agent(project_id: str, db) -> BeatToTimelineMapperResult:
    agent_key = "beat_to_timeline_mapper"
    run_start = time.time()
    logger.info(f"Initializing Agent [1]... | project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error("Agent [1]: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"[{agent_key}] Fetching data for project_id={project_id}")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")

        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        if not strategy_doc:
            raise ValueError(f"Strategy document for '{project_id}' not found")

        pipeline_doc = await db[PIPELINE_COLLECTION].find_one({"project_id": str(project_id)})
        if not pipeline_doc:
            logger.info(f"[{agent_key}] Pipeline document not found. Continuing with empty pipeline context.")
            pipeline_doc = {}

        script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
        if not script_doc:
            logger.info(f"[{agent_key}] Script document not found. Continuing with empty script context.")
            script_doc = {}

        phase_2_output = ideation_doc.get("phase_2_output", ideation_doc)
        approved_concepts = phase_2_output.get("approved_concepts", ideation_doc.get("approved_concepts", []))
        if not approved_concepts:
            raise ValueError("No approved concepts found in ideation document")

        concept_index = 0
        concept = approved_concepts[concept_index]
        concept_hook = concept.get("concept_hook")

        # Format-group-aware beat extraction
        format_group = phase_2_output.get("format_group", ideation_doc.get("format_group", "N"))
        is_visual = format_group == "V"

        if is_visual:
            # Group V: use composition_beats from the concept.
            # Fall back to visual_structure.composition_flow beat labels if composition_beats is empty.
            beats = concept.get("composition_beats", [])
            if not beats:
                visual_structure = phase_2_output.get("visual_structure", ideation_doc.get("visual_structure", {}))
                beats = [b.get("beat_label", "") for b in visual_structure.get("composition_flow", []) if b.get("beat_label")]
            beat_field_name = "composition_beats"
        else:
            beats = concept.get("story_beats", [])
            beat_field_name = "story_beats"

        narrative_budget = phase_2_output.get("narrative_budget", ideation_doc.get("narrative_budget", {}))
        scene_intelligence = phase_2_output.get("scene_intelligence", ideation_doc.get("scene_intelligence", {}))
        platform_rules = phase_2_output.get("platform_rules", ideation_doc.get("platform_rules", {}))
        video_type_final = phase_2_output.get("video_type_final", ideation_doc.get("video_type_final", ""))
        video_length_seconds = (
            project_doc.get("video_length_seconds")
            or phase_2_output.get("video_length_seconds")
            or ideation_doc.get("video_length_seconds")
        )

        number_of_shots = (
            project_doc.get("number_of_shots")
            or (int(video_length_seconds) // 8 if video_length_seconds else None)
        )
        if number_of_shots is None:
            raise ValueError("number_of_shots is missing from project document")

        if not concept_hook:
            raise ValueError("concept_hook is missing for the selected approved concept")
        if not beats:
            raise ValueError(f"{beat_field_name} are missing for the selected approved concept")
        if not narrative_budget:
            raise ValueError("narrative_budget is missing in ideation output")
        if video_length_seconds is None:
            raise ValueError("video_length_seconds is missing from project/ideation documents")

        placement_timing = scene_intelligence.get("placement_timing")
        selected_role = scene_intelligence.get("selected_role")

        logger.info(
            f"[{agent_key}] Extracted key inputs: concept_hook_present={bool(concept_hook)}, "
            f"beats_count={len(beats)} ({beat_field_name}), format_group={format_group}, "
            f"narrative_budget_keys={list(narrative_budget.keys())}, "
            f"placement_timing_present={placement_timing is not None}, selected_role_present={selected_role is not None}, "
            f"video_type_final_present={bool(video_type_final)}, video_length_seconds={video_length_seconds}, "
            f"number_of_shots={number_of_shots}"
        )

        prompt_input = {
            "concept": {
                "concept_hook": concept_hook,
                beat_field_name: beats,
            },
            "narrative_budget": {
                "total_seconds": narrative_budget.get("total_seconds"),
                "hook_seconds": narrative_budget.get("hook_seconds"),
                "tension_seconds": narrative_budget.get("tension_seconds"),
                "demo_seconds": narrative_budget.get("demo_seconds"),
                "payoff_seconds": narrative_budget.get("payoff_seconds"),
                "offer_seconds": narrative_budget.get("offer_seconds"),
                "compression_flags": narrative_budget.get("compression_flags", []),
            },
            "scene_intelligence": {
                "placement_timing": placement_timing,
                "selected_role": selected_role,
            },
            "platform_rules": {
                "hook_window_rule": platform_rules.get("hook_window_rule"),
                "soft_rules": platform_rules.get("soft_rules", []),
            },
            "video_type_final": video_type_final,
            "video_length_seconds": video_length_seconds,
            "number_of_shots": number_of_shots,
            "shot_duration_seconds": 8,
            "existing_pipeline_context": {
                "last_agent_logs_count": len(pipeline_doc.get("agent_logs", [])),
            },
            "existing_script_context": {
                "has_master_timeline": "master_timeline" in script_doc,
            },
        }
        prompt_input_json = json.dumps(prompt_input, indent=2)

        fixed_windows_list = "\n".join(
            f"  W{str(i + 1).zfill(2)}: {i * 8:.1f}s – {(i + 1) * 8:.1f}s"
            for i in range(number_of_shots)
        )

        hook_windows = int(narrative_budget.get('hook_seconds') or 0) // 8
        tension_windows = int(narrative_budget.get('tension_seconds') or 0) // 8
        demo_windows = int(narrative_budget.get('demo_seconds') or 0) // 8
        payoff_windows = int(narrative_budget.get('payoff_seconds') or 0) // 8
        offer_windows = int(narrative_budget.get('offer_seconds') or 0) // 8

        if is_visual:
            beat_section_label = "CONCEPT COMPOSITION BEATS (visual flow — no narrative arc):"
            beat_function_desc = "a beat_function describing the VISUAL job of that window in one sentence (what is shown, what changes, what the viewer experiences visually)"
            step1_instruction = f"Map the timing budget to window counts. The hook budget covers the first {hook_windows} window(s), visual build the next {tension_windows + demo_windows}, product reveal the next {payoff_windows}, and offer/CTA the final {offer_windows}. Use these counts as your starting label assignment."
            step2_instruction = f"Assign a beat_label and beat_function to each of the {number_of_shots} fixed windows. Every window must have a label drawn from the composition beats. No window may be left unlabeled."
            scene_constraint_note = f"Scene placement constraint (visual): {selected_role} at {placement_timing}" if selected_role and placement_timing else "No scene placement constraint."
        else:
            beat_section_label = "CONCEPT STORY BEATS:"
            beat_function_desc = "a beat_function describing the emotional/narrative job of that window in one sentence"
            step1_instruction = f"Map the narrative budget to window counts. The hook budget covers the first {hook_windows} window(s), tension the next {tension_windows}, demo the next {demo_windows}, payoff the next {payoff_windows}, and offer the final {offer_windows}. Use these counts as your starting label assignment."
            step2_instruction = f"Assign a beat_label and beat_function to each of the {number_of_shots} fixed windows. Every window must have a label drawn directly from the story beats. No window may be left unlabeled."
            scene_constraint_note = f"Selected role: {selected_role}\nMandatory placement window: {placement_timing}"

        prompt = f"""
You are a precision timeline architect for short-form video scripts.

Your sole function is structural — you assign beat labels to exactly {number_of_shots} pre-defined, fixed-duration time windows. You do not choose window lengths. You do not create windows. The {number_of_shots} windows are fixed at 8 seconds each. Your job is to decide which beat belongs in each window.

You are strictly grounded in the inputs provided. Do not introduce external assumptions. Your output is the structural backbone for all downstream agents — errors here cascade through the entire pipeline.

You are labeling the master timeline for a {video_length_seconds}-second {video_type_final} video ({number_of_shots} shots × 8 seconds each).

FIXED WINDOWS (start_s and end_s are non-negotiable — do NOT change them):
{fixed_windows_list}

TIMING BUDGET:
Total duration: {narrative_budget.get('total_seconds')}s ({number_of_shots} shots)
- Hook/Opening: {narrative_budget.get('hook_seconds')}s → {hook_windows} window(s)
- Build/Tension: {narrative_budget.get('tension_seconds')}s → {tension_windows} window(s)
- Demo/Reveal: {narrative_budget.get('demo_seconds')}s → {demo_windows} window(s)
- Payoff: {narrative_budget.get('payoff_seconds')}s → {payoff_windows} window(s)
- Offer/CTA: {narrative_budget.get('offer_seconds')}s → {offer_windows} window(s)
Compression flags: {narrative_budget.get('compression_flags', [])}

{beat_section_label}
{beats}

CONCEPT HOOK (opening — must be assigned to the first window):
{concept_hook}

SCENE PLACEMENT CONSTRAINT:
{scene_constraint_note}

PLATFORM HOOK RULE:
{platform_rules.get('hook_window_rule')}

VIDEO TYPE:
{video_type_final}

TASK:
Step 1 — {step1_instruction}

Step 2 — {step2_instruction}

Step 3 — Honor the scene placement constraint if present. The window whose time range contains the midpoint of the mandatory placement must carry the correct beat_label. Adjust adjacent labels if needed without adding or removing windows.

Step 4 — Label each window with: window ID (W01, W02, ..., W{str(number_of_shots).zfill(2)}), start_s (fixed), end_s (fixed), a beat_label drawn from the beats, and {beat_function_desc}.

ABSOLUTE CONSTRAINTS:
- You must output exactly {number_of_shots} window objects — no more, no fewer.
- start_s and end_s are fixed. W01 is always 0.0–8.0, W02 is 8.0–16.0, and so on. Do NOT alter these values.
- Do not add fractional windows. Do not subdivide windows. Each window is exactly 8 seconds.
- No creative content. No VO copy. Beat labels and functions only.

Return only valid JSON matching the schema exactly.
Do not include markdown, prose preamble, or code fences.
Do not add extra fields or omit required fields.
Keep reasoning, rationale, and timing_decisions concise but explicit.
"""

        invoke_start = time.time()
        logger.info(f"Agent [1]: Preparing to call Gemini model={GEMINI_MODEL}...")
        try:
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
            )
            logger.info("Agent [1]: Gemini Client instantiated. Sending prompt...")

            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": BeatToTimelineMapperResult.model_json_schema(),
                    "automatic_function_calling": {"disable": True},
                }
            )

            api_duration = time.time() - invoke_start
            logger.info(f"Agent [1]: Gemini API call completed in {api_duration:.2f}s")
            logger.debug(f"Agent [1]: Raw response length={len(response.text)} chars")

            cleaned_json = _clean_json_string(response.text)
            parsed_data = json.loads(cleaned_json)
            logger.info("Agent [1]: Successfully parsed JSON response.")

            result = BeatToTimelineMapperResult(**parsed_data)
            result.status = "completed"
            logger.info("Agent [1]: Successfully validated structured output with Pydantic.")
        except Exception as llm_error:
            if "503" in str(llm_error) or "429" in str(llm_error):
                logger.error(f"Gemini API overloaded: {llm_error}")
            else:
                logger.error(f"Agent [1]: Gemini invocation failed: {llm_error}", exc_info=True)
            raise

        logger.info("Agent [1]: Updating SCRIPT and PIPELINE collections...")
        await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {
                "$set": {
                    "project_id": str(project_id),
                    "master_timeline": result.master_timeline.model_dump(),
                }
            },
            upsert=True,
        )

        pipeline_log = {
            "agent_id": 1,
            "agent_name": "beat_to_timeline_mapper",
            "execution_time_utc": time.time(),
            "api_duration_s": round(api_duration, 2),
            "duration_s": round(time.time() - run_start, 2),
            "reasoning": result.reasoning,
            "beat_expansion_log": [item.model_dump() for item in result.beat_expansion_log],
            "rationale": result.rationale,
            "timing_decisions": result.timing_decisions,
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )

        total_duration = time.time() - run_start
        logger.info("Agent [1]: Successfully updated SCRIPT and PIPELINE collections.")
        logger.info(f"Agent [1]: Complete! Total duration {total_duration:.2f}s")
        return result

    except Exception as e:
        logger.error(f"[{agent_key}] Failed with error: {str(e)}", exc_info=True)
        raise
