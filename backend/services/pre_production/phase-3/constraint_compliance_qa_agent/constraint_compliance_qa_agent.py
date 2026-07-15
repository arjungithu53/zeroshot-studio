import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------------
# GLOBALS & CONFIGURATION
# -----------------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")

logger = logging.getLogger("zeroshot.phase3.constraint_compliance_qa_agent")

def _clean_json_string(raw: str) -> str:
    """Helper to remove markdown fencing if the model hallucinates it."""
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

# -----------------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------------

class Tier1Result(BaseModel):
    constraint: str
    pass_: bool = Field(alias="pass")

class Tier2Result(BaseModel):
    constraint: str
    severity: str

class Tier3Result(BaseModel):
    anchor: str
    present: bool

class Tier4Result(BaseModel):
    conflict_flag: str
    resolved: bool

class RevisionReportFailure(BaseModel):
    constraint: str
    target_agent: str
    fix: str

class RevisionReport(BaseModel):
    failures: List[RevisionReportFailure] = Field(default_factory=list)

class QaResultModel(BaseModel):
    status: str
    tier_1_results: List[Tier1Result]
    tier_2_results: List[Tier2Result]
    tier_3_results: List[Tier3Result]
    tier_4_results: List[Tier4Result]
    revision_report: Optional[RevisionReport] = None

# -----------------------------------------------------------------------------------
# AGENT RUNNER
# -----------------------------------------------------------------------------------
async def run_constraint_compliance_qa_agent(project_id: str, db: Any) -> QaResultModel:
    agent_id = 10
    agent_key = "constraint_compliance_qa_agent"
    run_start = time.time()
    logger.info(f"Initializing Agent {agent_id} ({agent_key})... | project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error(f"Agent {agent_id}: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"Agent {agent_id}: Fetching data for project_id={project_id}")

        # Fetch Docs
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")
            
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")

        script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
        if not script_doc:
            raise ValueError(f"Script document for '{project_id}' not found")

        # Extract Phase 2 Output
        phase_2_output = ideation_doc.get("phase_2_output", {})
        constraint_graph = phase_2_output.get("constraint_graph", {})
        hard_constraints = constraint_graph.get("hard_constraints", [])
        soft_constraints = constraint_graph.get("soft_constraints", [])
        conflict_flags = constraint_graph.get("conflict_flags", [])

        # Get specific concept — fall back to first approved concept if concept_id not set in script_doc
        concept_id = script_doc.get("concept_id")
        approved_concepts = phase_2_output.get("approved_concepts", [])
        if concept_id:
            concept = next((c for c in approved_concepts if c.get("concept_id") == concept_id), {})
        else:
            concept = approved_concepts[0] if approved_concepts else {}
        
        constraint_anchors = concept.get("constraint_anchors", [])

        format_group = phase_2_output.get("format_group", ideation_doc.get("format_group", "N"))
        is_visual = format_group == "V"
        if is_visual:
            selected_motif = phase_2_output.get("selected_visual_motif", ideation_doc.get("selected_visual_motif", {}))
            archetype = selected_motif.get("selected_motif", concept.get("visual_motif", ""))
            micro_policy = selected_motif.get("visual_micro_policy", "")
            failure_modes = selected_motif.get("failure_modes", [])
        else:
            selected_archetype = phase_2_output.get("selected_archetype", {})
            archetype = concept.get("archetype", "")
            micro_policy = selected_archetype.get("micro_policy", "")
            failure_modes = selected_archetype.get("failure_modes", [])

        brand_guardrails = ideation_doc.get("brand_guardrails", {})

        target_audience = project_doc.get("target_audience", {})
        number_of_shots = project_doc.get("number_of_shots")
        offer_constraints = script_doc.get("offer_constraints", {})

        # Script Data
        master_timeline = script_doc.get("master_timeline", {})
        shot_list = script_doc.get("shot_list", {})
        av_channel_map = script_doc.get("av_channel_map", {})
        vo_script = script_doc.get("vo_script", {})
        dialogue_lines = script_doc.get("dialogue_lines", {})
        audio_design = script_doc.get("audio_design", {})
        pacing_map = script_doc.get("pacing_map", {})
        loop_revision_requests = script_doc.get("loop_revision_requests", {})

        logger.info(f"Agent {agent_id}: Extracted script execution state and Phase 2 outputs.")

        prompt = f"""You are the contract enforcer for the Phase 3 scriptwriting pipeline. You receive the fully assembled script after all creative and post-processing agents have run, and you run it against every constraint established in Phase 2. You do not fix anything. You identify failures, assign them to the responsible target_agent, and specify exactly what needs to change.

You evaluate across four tiers in strict order: hard constraints (binary pass/fail), soft constraints (holistic evaluation), concept-specific constraint anchors, and conflict flag resolutions. A green light from you sends the script to formatting. A fail produces a structured revision report.

You are grounded strictly in the assembled script and Phase 2 constraints provided. Perform all evaluations based solely on what is present in the script — do not infer, assume, or give benefit of the doubt.

You are performing QA on the assembled script for concept: {concept.get('concept_id', 'Unknown')}

PHASE 2 HARD CONSTRAINTS:
{json.dumps(hard_constraints, indent=2)}

PHASE 2 SOFT CONSTRAINTS:
{json.dumps(soft_constraints, indent=2)}

PHASE 2 CONFLICT FLAGS TO RESOLVE:
{json.dumps(conflict_flags, indent=2)}

CONCEPT-SPECIFIC CONSTRAINT ANCHORS:
{json.dumps(constraint_anchors, indent=2)}

CONCEPT ARCHETYPE:
{archetype}

ARCHETYPE MICRO-POLICY:
{micro_policy}
Failure modes: {json.dumps(failure_modes, indent=2)}

BRAND GUARDRAILS:
Tonal guardrails: {json.dumps(brand_guardrails.get('tonal_guardrails', []), indent=2)}
Mandatory implications: {json.dumps(brand_guardrails.get('mandatory_implications', []), indent=2)}

TARGET AUDIENCE:
{json.dumps(target_audience, indent=2)}

OFFER CONSTRAINTS:
{json.dumps(offer_constraints, indent=2)}

STRUCTURAL REQUIREMENTS — check these before all tiers:
- The master timeline must contain exactly {number_of_shots} windows.
- Each window must be exactly 8.0 seconds (end_s - start_s must equal 8.0 for every window).
- The shot_list must contain exactly {number_of_shots} entries (one per window).
If any of these structural counts differ from {number_of_shots}, issue a TIER 1 HARD FAILURE with:
  target_agent: "agent_1"
  constraint: "structural_shot_count"
  fix: "Expected {number_of_shots} windows of exactly 8.0s each. Got [actual count]. Beat to Timeline Mapper must be re-run."

ASSEMBLED SCRIPT (full — all channels):
Master timeline: {json.dumps(master_timeline.get('windows', []), indent=2)}
Shot list: {json.dumps(shot_list, indent=2)}
AV channel map: {json.dumps(av_channel_map, indent=2)}
VO script: {json.dumps(vo_script, indent=2)}
Dialogue lines: {json.dumps(dialogue_lines, indent=2)}
Audio design: {json.dumps(audio_design, indent=2)}
Pacing map: {json.dumps(pacing_map, indent=2)}
Loop revision requests (outstanding): {json.dumps(loop_revision_requests, indent=2)}

TASK — evaluate in strict tier order:

TIER 1 — Hard constraints (binary pass/fail). Check each hard constraint against the assembled script. Any failure is mandatory — it stops the script from proceeding. For each failure: state the constraint, identify which specific part of the script fails it, identify which target_agent is responsible, and state exactly what must change.

TIER 2 — Soft constraints (holistic evaluation). Evaluate whether the assembled script as a whole satisfies the tone, primary message, and support point requirements. This requires reading the VO script and shot list together. Rank any failures by severity (high/medium/low).

TIER 3 — Concept-specific constraint anchors. Each anchor in concept.constraint_anchors is a strategic requirement unique to this concept. Verify each is present and correctly executed in the assembled script. A concept that loses its distinctive strategic angle fails this tier.

TIER 4 — Conflict flag resolution. Phase 2 flagged specific conflicts that downstream agents were required to resolve. Verify each was resolved correctly. Example: "Strawberry and Pomegranate must use mythic language" — check whether the VO script uses elevated language ("Wild Himalayan Strawberry") or defaulted to grocery register ("strawberry and pomegranate").

CONSTRAINTS (apply last):
- Do not perform repairs or suggest creative alternatives. Identify and assign only.
- If any loop_revision_requests are outstanding (non-null), flag them as Tier 1 failures — the loop must be resolved before the script is approved.
- When assigning failures, `target_agent` MUST be strictly formatted as exactly one of the following strings according to the responsible pipeline agent:
  - "agent_1" (Beat to Timeline Mapper - structural failure, pipeline must restart from Agent 1)
  - "agent_3" (Visual Sequencing Agent - fixes shots/visuals)
  - "agent_4" (AV Separation Agent - fixes shot pacing/windows)
  - "agent_5" (Voiceover Writer Agent - fixes VO lines/tone)
  - "agent_6" (Dialogue Agent - fixes character dialogue)
  - "agent_7" (Audio Design Agent - fixes SFX/BGM)
  - "agent_8" (Rhythm & Pacing Regulator - fixes timeline pacing/timing)
  - "agent_9" (Loop Optimization Agent - fixes loop transitions/hooks)
- Output as a JSON object with fields: status ("pass" or "fail"), tier_1_results (array), tier_2_results (array), tier_3_results (array), tier_4_results (array), revision_report (object with failures array, each containing: constraint, target_agent, fix — or null if status is pass)."""

        invoke_start = time.time()
        logger.info(f"Agent {agent_id}: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent {agent_id}: Gemini Client instantiated. Sending prompt...")

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": QaResultModel.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )

        api_duration = time.time() - invoke_start
        logger.info(f"Agent {agent_id}: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent {agent_id}: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent {agent_id}: Successfully parsed JSON response.")

        result = QaResultModel(**parsed_data)
        logger.info(f"Agent {agent_id}: Successfully validated structured output with Pydantic.")

        logger.info(f"Agent {agent_id}: Updating SCRIPT and PIPELINE collections...")
        result_dict = result.model_dump(by_alias=True)
        
        qa_status_label = 'pass' if result.status == 'pass' \
                          else 'pass_with_warnings'

        await db[SCRIPT_COLLECTION].update_one(
            {'project_id': str(project_id)},
            {'$set': {
                'qa_result':       result_dict,
                'qa_status_label': qa_status_label,  # 'pass' or 'pass_with_warnings'
            }},
            upsert=True
        )

        # Log the qa_status_label clearly
        logger.info(f'Agent 10: QA status label = {qa_status_label} for project_id={project_id}')

        pipeline_log = {
            "agent_id": agent_id,
            "agent_name": agent_key,
            "execution_time_utc": time.time(),
            "api_duration_s": round(api_duration, 2),
            "duration_s": round(time.time() - run_start, 2),
            "status": getattr(result, 'status', 'completed'),
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        total_duration = time.time() - run_start
        logger.info(f"Agent {agent_id}: Complete! Total duration {total_duration:.2f}s")
        return result

    except Exception as e:
        logger.error(f"[{agent_key}] Failed with error: {str(e)}", exc_info=True)
        raise
