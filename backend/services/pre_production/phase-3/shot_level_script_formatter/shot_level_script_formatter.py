import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import csv, io

import boto3
from dotenv import load_dotenv
from bson import ObjectId
from pydantic import BaseModel, Field
from google import genai

logger = logging.getLogger("zeroshot.phase3.shot_level_script_formatter")

def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

# Load environment variables globally
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "zeroshot-v1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")

class ScriptScene(BaseModel):
    scene_number:  int   = Field(description="1-indexed scene number, increments on beat_label change")
    scene_name:    str   = Field(description="beat_label shared by all windows in this scene group")
    scene_script:  str   = Field(description="Full Fountain content for all windows in this scene")

class ScriptFormatterOutput(BaseModel):
    scenes:                        List[ScriptScene]
    channel_density_map:           str = Field(description="Per-window active channel log")
    constraint_compliance_summary: str = Field(description="Compliance checklist")
    reasoning:                     str = Field(description="Flags and grouping notes")
    fountain_s3_url:               str = Field(default="")
    status:                        str = "pending"

async def run_shot_level_script_formatter(project_id: str, db, ideation_id: str = "") -> ScriptFormatterOutput:
    agent_key = "shot_level_script_formatter"
    run_start = time.time()
    logger.info(f"Initializing Agent 11: {agent_key} for project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error(f"Agent 11 [{agent_key}]: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"Agent 11 [{agent_key}]: Fetching data for project_id={project_id}")
        
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")
            
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")

        script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
        if not script_doc:
            raise ValueError(f"Script document for '{project_id}' not found")
            
        pipeline_doc = await db[PIPELINE_COLLECTION].find_one({"project_id": str(project_id)})
        
        # Get concept and ideation info
        if not ideation_id:
            ideation_id = str(ideation_doc.get("_id", ""))
        
        # Determine current concept (first one logic as standard)
        phase_2_output = ideation_doc.get("phase_2_output", ideation_doc)
        approved_concepts = phase_2_output.get("approved_concepts", ideation_doc.get("approved_concepts", []))
        concept_data = {}
        if approved_concepts and len(approved_concepts) > 0:
            concept_data = approved_concepts[0]
            
        # Get upstream channel streams from script doc
        master_timeline = script_doc.get("master_timeline", {})
        shot_list = script_doc.get("shot_list", [])
        av_channel_map = script_doc.get("av_channel_map", [])
        vo_script = script_doc.get("voiceover_script", script_doc.get("vo_script", {}))
        dialogue_lines = script_doc.get("dialogue_lines", [])
        audio_design = script_doc.get("audio_design", {})
        pacing_map = script_doc.get("rhythm_pacing_map", script_doc.get("pacing_map", {}))
        loop_revision_requests = script_doc.get("loop_revision_requests", [])
        
        logger.info(f"Agent 11 [{agent_key}]: Successfully extracted inputs from DB docs.")

        prompt = f"""
You are a pure assembly agent. You make zero creative decisions. Every upstream agent has already made all the creative and strategic choices. Your job is to take their outputs — which currently exist as separate channel data streams — and serialize them into a single, unified, time-indexed Fountain format document that a production team (director, cinematographer, editor, sound designer) can execute from directly.

Every entry you produce must be production-actionable. If you encounter a shot description that reads like a creative brief rather than a direction, you flag it and do not include it in the formatted output. Your output is a valid .fountain plain-text file and two supplementary outputs: a channel density map and a constraint compliance summary.

You are grounded strictly in the assembled and QA-cleared script provided.

You are formatting the final production script for concept: {concept_data.get('concept_id', 'Concept 1')}

IDEATION ID: {ideation_id}
PROJECT ID: {project_id}

QA STATUS: PASS (this script has cleared Agent 10)

ASSEMBLED SCRIPT CHANNELS:
Master timeline: {json.dumps(master_timeline.get('windows', []) if isinstance(master_timeline, dict) else master_timeline, indent=2)}
Shot list: {json.dumps(shot_list, indent=2)}
AV channel map: {json.dumps(av_channel_map, indent=2)}
VO script: {json.dumps(vo_script, indent=2)}
Dialogue lines: {json.dumps(dialogue_lines, indent=2)}
Audio design windows: {json.dumps(audio_design.get('windows', []) if isinstance(audio_design, dict) else audio_design, indent=2)}
Music mood curve: {json.dumps(audio_design.get('music_mood_curve', '') if isinstance(audio_design, dict) else '', indent=2)}
Pacing map: {json.dumps(pacing_map.get('windows', []) if isinstance(pacing_map, dict) else pacing_map, indent=2)}
Loop revision notes (for annotation): {json.dumps(loop_revision_requests, indent=2)}

TASK:
Produce a scene-by-scene script breakdown. Output a scenes array.
One entry per scene — NOT one entry per window.

SCENE GROUPING RULE:
  Windows sharing the same beat_label belong to the same scene.
  scene_number increments only when beat_label changes from the
  previous window. Start at scene_number = 1.

  Example:
    W01 beat_label = "Intrigue to Deprivation" → scene_number = 1
    W02 beat_label = "Intrigue to Deprivation" → scene_number = 1
    W03 beat_label = "Ritual to Exaltation"    → scene_number = 2
    W04 beat_label = "Ritual to Exaltation"    → scene_number = 2
    W08 beat_label = "Offering"                → scene_number = 3

  scene_name = the shared beat_label of that group.

FOR EACH scene entry, build scene_script by concatenating the
Fountain content of ALL windows in that scene group in window
order. For each window within the scene, append the following
in this exact order with no labels or headings between them:

  .{{window_id}} — {{start_s}}s-{{end_s}}s — {{beat_label}}

  # {{beat_label}}
  (only on the FIRST window of the scene group —
  omit entirely on all subsequent windows in the same scene)

  {{location_heading field from shot_list for this window_id}}
  (e.g. INT. CORPORATE BREAKROOM - DAY. Write on every window.
  Update it when the location changes between windows in the
  same scene. This is a standard Fountain scene heading.)

  {{Narrative action block — write as proper screenplay prose:
    • If character_intro is non-empty for this window_id, embed
      it into the opening action sentence:
        "CREATOR (late 20s, heavy-lidded eyes, beige blazer)
        drops a yellow chapstick tube past the lens with a sigh."
    • For subsequent windows where character_intro is empty,
      write the action as narrative prose using the subject,
      action, and notes fields as source material.
    • Translate all framing and lighting details into atmospheric
      scene prose that implies the visual — do NOT output raw
      camera specs. No lens mm values, no "ECU", no "MCU", no
      "handheld snap-zoom", no cinematography jargon.
    • Preserve all meaningful visual detail: texture, color,
      product moments, emotional beats, set atmosphere.
    • Write in present tense. Each action paragraph should read
      like a produced screenplay, not a shot brief.
    • If shot data is absent or non-actionable:
      write [FLAGGED: requires Agent 3 revision]}}

  _{{TEXT SUPER: content}}_
  (one line per text super — only if text_supers exist
  for this window in the shot_list. Omit entirely if none.)

  CREATOR (V.O.)
  {{vo line from vo_script for this window_id}}
  (if window is silent: write [SILENT] as the dialogue line)

  {{CHARACTER NAME}}
  {{dialogue line}}
  (only if dialogue_lines contains entries for this window_id.
  Omit entirely if no dialogue for this window.)

  [[LOOP NOTE: {{instruction}}]]
  (only if loop_revision_requests contains an entry targeting
  this window. Omit entirely if none.)

After all scene entries, also output:
- channel_density_map: plain-text table, one row per window
  showing which channels are simultaneously active:
  VO / TEXT / SFX / MUSIC
- constraint_compliance_summary:
  /* CONSTRAINT COMPLIANCE SUMMARY
  Tier 1: [each hard constraint with window where satisfied]
  Tier 2: [each soft constraint with where/how satisfied]
  Tier 3: [each anchor with where satisfied]
  */
- reasoning: brief notes on scene grouping decisions and any
  flagged shots

CONSTRAINTS (apply last):
- scene_script must be valid Fountain plain-text with no labels,
  headings, or metadata — only Fountain elements
- The # section marker appears exactly once per scene, on the
  first window of that beat group only
- scene_number starts at 1 and increments only on beat_label
  change — never once per window
- No window may be skipped — every window must appear in
  exactly one scene's scene_script
- Output valid JSON matching the schema exactly
"""

        invoke_start = time.time()
        logger.info(f"Agent 11 [{agent_key}]: Preparing to call Gemini model={GEMINI_MODEL}...")

        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 11 [{agent_key}]: Gemini Client instantiated. Sending prompt...")

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ScriptFormatterOutput.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )

        api_duration = time.time() - invoke_start
        logger.info(f"Agent 11 [{agent_key}]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 11 [{agent_key}]: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 11 [{agent_key}]: Successfully parsed JSON response.")

        result = ScriptFormatterOutput(**parsed_data)
        result.status = "completed"
        
        # Build CSV from scenes
        csv_output = io.StringIO()
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        writer.writerow(['scene_number', 'scene_name', 'scene_script'])
        for scene in result.scenes:
            writer.writerow([
                scene.scene_number,
                scene.scene_name,
                scene.scene_script
            ])
        csv_content = csv_output.getvalue()

        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            try:
                s3_client = boto3.client(
                    's3',
                    region_name=AWS_REGION,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                )
                concept_name = concept_data.get('concept_id', 'concept_1').replace(" ", "_")
                file_key = f"scripts/{project_id}/{concept_name}.csv"

                logger.info(f"Agent 11 [{agent_key}]: Uploading script CSV to S3 at '{file_key}'...")
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=file_key,
                    Body=csv_content.encode('utf-8'),
                    ContentType='text/csv'
                )
                result.fountain_s3_url = f"s3://{S3_BUCKET_NAME}/{file_key}"
                logger.info(f"Agent 11 [{agent_key}]: Successfully uploaded to {result.fountain_s3_url}")
            except Exception as s3_err:
                logger.error(f"Agent 11 [{agent_key}]: S3 upload failed: {s3_err}")
                result.fountain_s3_url = f"UPLOAD_FAILED: {s3_err}"
        else:
            logger.warning(f"Agent 11 [{agent_key}]: Missing AWS credentials, skipping S3 upload.")
            concept_name = concept_data.get('concept_id', 'concept_1').replace(" ", "_")
            result.fountain_s3_url = (
                f"s3://{S3_BUCKET_NAME}/scripts/{project_id}/{concept_name}.csv"
            )
        
        logger.info(f"Agent 11 [{agent_key}]: Successfully validated structured output with Pydantic.")

        # Database Updates
        logger.info(f"Agent 11 [{agent_key}]: Updating SCRIPT and PIPELINE collections...")
        
        update_result = await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "formatted_script": {
                    "script_s3_url":                 result.fountain_s3_url,
                    "scenes":                        [s.model_dump() for s in result.scenes],
                    "channel_density_map":           result.channel_density_map,
                    "constraint_compliance_summary": result.constraint_compliance_summary,
                }
            }}
        )
        
        if update_result.modified_count == 0:
            logger.warning(f"Agent 11 [{agent_key}]: No documents modified in script collection. Upserting is handled implicitly if needed or project did not change.")

        total_duration = time.time() - run_start
        pipeline_log = {
            "agent_id": 11,
            "agent_name": agent_key,
            "status": "completed",
            "execution_time_sec": total_duration,
            "timestamp": datetime.utcnow().isoformat(),
            "reasoning": result.reasoning
        }
        
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        logger.info(f"Agent 11 [{agent_key}]: Total duration {total_duration:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Agent 11 [{agent_key}] runtime error: {str(e)}", exc_info=True)
        raise e
