import os
import csv
import io
import json
import time
import logging
from datetime import datetime
from typing import List

import boto3
from dotenv import load_dotenv
from bson import ObjectId
from pydantic import BaseModel, Field
from google import genai

logger = logging.getLogger("zeroshot.phase3.final_shotlist_agent")

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
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")

class ShotRow(BaseModel):
    scene_number:    int
    shot_number:     str   # e.g. "1.1", "2.3"
    shot_type:       str
    camera_movement: str
    description:     str
    characters:      str   # comma-separated or empty string
    locations:       str
    product_present: str   # "Yes" or "No" only

class FinalShotListOutput(BaseModel):
    shots:           List[ShotRow]
    reasoning:       str = Field(description="Notes on shot breakdown decisions")
    shotlist_s3_url: str = Field(default="")
    status:          str = "pending"

async def run_final_shot_agent(project_id: str, db) -> FinalShotListOutput:
    agent_key = "final_shotlist_agent"
    run_start = time.time()
    logger.info(f"Initializing Agent 12: {agent_key} for project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error(f"Agent 12 [{agent_key}]: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"Agent 12 [{agent_key}]: Fetching data for project_id={project_id}")
        
        project_doc = await db[PROJECTS_COLLECTION].find_one({'_id': ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")
            
        ideation_doc = await db[IDEATION_COLLECTION].find_one({'project_id': str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")

        script_doc = await db[SCRIPT_COLLECTION].find_one({'project_id': str(project_id)})
        if not script_doc:
            raise ValueError(f"Script document for '{project_id}' not found")

        # From script_doc
        formatted_script = script_doc.get('formatted_script', {})
        scenes           = formatted_script.get('scenes', [])
        shot_list        = script_doc.get('shot_list', [])
        master_timeline  = script_doc.get('master_timeline', {})
        windows          = master_timeline.get('windows', []) if isinstance(master_timeline, dict) else []

        # From ideation_doc
        phase_2_output   = ideation_doc.get('phase_2_output', {})
        approved_concepts = phase_2_output.get('approved_concepts', [])
        concept          = approved_concepts[0] if (approved_concepts and len(approved_concepts) > 0) else {}
        concept_id       = concept.get('concept_id', 'concept_1')
        
        # From project_doc
        product_details  = project_doc.get('product_details', '')
        number_of_shots  = project_doc.get('number_of_shots')

        prompt = f"""
You are a production shot list breakdown agent. You read a
Fountain-formatted script and a shot list and produce a
production-ready CSV shot list with one row per individual
camera setup.

PRODUCT BEING ADVERTISED:
{product_details}

MASTER TIMELINE (for scene grouping and timestamps):
{json.dumps(windows, indent=2)}

SCRIPT SCENES (Fountain format, one entry per beat group):
{json.dumps(scenes, indent=2)}

DETAILED SHOT LIST (framing, action, lighting per window):
{json.dumps(shot_list, indent=2)}

TASK:
Produce a shots array — one ShotRow per window (each window = one 8-second camera setup).

SCENE GROUPING RULE:
  Windows sharing the same beat_label belong to the same scene.
  scene_number increments only when beat_label changes.
  Start at scene_number = 1.

SHOT NUMBERING RULE:
  shot_number uses decimal notation. Integer part = scene_number.
  Decimal part increments per shot within that scene, one shot per window.
  Example: if scene 1 spans W01 and W02: 1.1 (W01), 1.2 (W02).

SHOTS PER WINDOW RULE (each shot = 8 seconds = one row):
  Produce exactly ONE ShotRow per window — no more, no fewer.
  Each window is a complete 8-second camera setup. Do NOT split a
  window into multiple rows. If the notes mention a camera movement
  (e.g., "push in", "snap zoom"), it describes movement within that
  single setup — encode it in camera_movement and keep it as one row.
  Total output: exactly {number_of_shots} ShotRow entries.

FOR EACH ShotRow:

shot_type — derive from framing field:
  ECU / Extreme Close Up / Macro -> extreme_close_up
  CU / Close Up                  -> close_up
  MCU / Medium Close Up          -> medium_shot
  MS / Medium Shot               -> medium_shot
  WS / Wide / LS                 -> wide_shot
  POV                            -> pov
  OTS / Over the Shoulder        -> over_the_shoulder

camera_movement — derive from notes field:
  snap zoom / whip pan           -> snap_zoom
  push in / slow push / dolly in -> push_in
  pull out / dolly out           -> pull_out
  slow pan / pan                 -> slow_pan
  handheld / drift               -> handheld
  locked off / locked            -> locked_off
  static / no movement specified -> static
  dolly (without direction)      -> dolly

description — one production-actionable sentence combining
  subject + action + lighting from the shot_list entry.
  No vague language. No creative briefs.
  If vague: [FLAGGED: requires Agent 3 revision]

characters — extract character names from subject or action
  fields. Comma-separate multiples. Empty string if none.

locations — short label derived from the environmental context
  in notes or action e.g. "Office Breakroom", "Bathroom Vanity",
  "Studio Tabletop", "Outdoor Rooftop".

product_present — "Yes" if the product (the item described in
  PRODUCT BEING ADVERTISED above) is physically visible or being
  handled in this shot based on the shot description.
  "No" if it is not present in frame.
  Only two values are valid: Yes or No.

CONSTRAINTS (apply last):
- Output must contain exactly {number_of_shots} ShotRow entries — one per window.
- shot_number must follow strict decimal notation, no gaps
- scene_number starts at 1, increments only on beat_label change
- shot_type and camera_movement must use only the exact enum
  values listed above — no other values permitted
- product_present must be exactly "Yes" or "No" — nothing else
- No shot may have an empty description
- Output valid JSON matching the schema exactly
"""
        
        logger.info(f"Agent 12 [{agent_key}]: Calling Gemini API...")
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": FinalShotListOutput.model_json_schema(),
                "automatic_function_calling": {"disable": True},
                "temperature": 0.2,
            }
        )

        raw_text = _clean_json_string(response.text)
        parsed_data = json.loads(raw_text)
        result = FinalShotListOutput(**parsed_data)
        
        logger.info(f"Agent 12 [{agent_key}]: Gemini call successful. Produced {len(result.shots)} shots.")

        # Build CSV from shots
        csv_output = io.StringIO()
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        writer.writerow([
            'scene_number', 'shot_number', 'shot_type',
            'camera_movement', 'description', 'characters',
            'locations', 'product_present'
        ])
        for shot in result.shots:
            writer.writerow([
                shot.scene_number, shot.shot_number, shot.shot_type,
                shot.camera_movement, shot.description,
                shot.characters, shot.locations, shot.product_present
            ])
        csv_content = csv_output.getvalue()

        # Upload to S3
        file_key = f"scripts/{project_id}/{concept_id}_shotlist.csv"
        
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME:
            try:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION
                )
                
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=file_key,
                    Body=csv_content.encode('utf-8'),
                    ContentType='text/csv'
                )
                s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_key}"
                result.shotlist_s3_url = s3_url
                logger.info(f"Agent 12 [{agent_key}]: Successfully uploaded to S3 -> {s3_url}")
            except Exception as e:
                logger.warning(f"Agent 12 [{agent_key}]: Failed to upload to S3: {e}")
                result.shotlist_s3_url = f"mock-s3-path://{file_key}"
        else:
            logger.warning(f"Agent 12 [{agent_key}]: AWS credentials missing. Skipping real S3 upload.")
            result.shotlist_s3_url = f"mock-s3-path://{file_key}"

        result.status = "completed"

        logger.info(f"Agent 12 [{agent_key}]: Updating MongoDB...")
        await db[SCRIPT_COLLECTION].update_one(
            {'project_id': str(project_id)},
            {'$set': {
                'final_shot_list': {
                    'shotlist_s3_url': result.shotlist_s3_url,
                    'shots':           [s.model_dump() for s in result.shots],
                }
            }}
        )

        run_time = time.time() - run_start
        log_entry = {
            "agent_id": agent_key,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "execution_time_seconds": round(run_time, 2),
            "output_summary": f"Generated {len(result.shots)} shot rows.",
            "shotlist_s3_url": result.shotlist_s3_url
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": log_entry}},
            upsert=True
        )

        logger.info(f"Agent 12 [{agent_key}]: Completed successfully in {run_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Agent 12 [{agent_key}]: Failed with error: {str(e)}")
        
        # Attempt to log failure to pipeline
        run_time = time.time() - run_start
        try:
            error_log = {
                "agent_id": agent_key,
                "status": "failed",
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time_seconds": round(run_time, 2),
                "error": str(e)
            }
            await db[PIPELINE_COLLECTION].update_one(
                {"project_id": str(project_id)},
                {"$push": {"agent_logs": error_log}},
                upsert=True
            )
        except Exception as log_error:
            logger.error(f"Agent 12 [{agent_key}]: Also failed to write error log to MongoDB: {log_error}")
            
        raise

