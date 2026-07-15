import os
import json
import time
import uuid
import logging
import tempfile
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

import boto3
from pymongo import MongoClient
from google import genai
from google.genai import types

from app.services.final_assemblies_service import update_agent_output, append_versioned
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.gemini_files import wait_for_gemini_file_active
from app.services.phase_4_agents.workflow_state import Phase4State

logger = logging.getLogger(__name__)

__all__ = ["EDLGeneratorAgent", "agent_1_edl_node"]

DR_EDITOR_SYSTEM_PROMPT = """You are an Expert DR (Direct Response) Video Editor specializing in high-converting, short-form ads for TikTok, Instagram Reels, and YouTube Shorts. Objective: I will provide you with a Script, a Shot List, and descriptions (or visual access) to Raw Video Clips. Your job is to create a highly engaging, fast-paced Edit Decision List (EDL) that pieces these raw files together into a final ad. CRITICAL RULES YOU MUST FOLLOW:

1. Visual Reality > Script Theory: NEVER blindly trust the script's directions if they contradict the actual provided footage. You must analyze the literal visual reality of the clips provided. Example: If the script calls for a "seamless match-on-action loop," you MUST verify that the final frame perfectly matches the first frame in camera angle, posture, lighting, and framing. If it does not, explicitly state that a seamless loop is impossible and recommend a Hard Cut/Hard Reset instead.
2. Flexible Timing & Ruthless Trimming (Pacing is Everything): Do not restrict yourself to the exact time lengths mentioned in the script. Your priority is hook rate, momentum, and viewer retention. Keep it punchy: Social media ads need to move fast. Do not linger on dead space. Skip Redundancy: Ruthlessly suggest skipping scenes or raw clips if they slow down the narrative or if two clips show the exact same action. Audio/Visual Balance: While pacing must be fast, ensure clips are left just long enough (typically 2 to 5 seconds) for the viewer to register the visual and for the Voiceover to realistically play out. Do not make the edits so fast that the ad feels glitchy.
3. Provide Exact Timestamps based on Visual Action: For every clip you decide to keep, you must provide the exact Trim IN and Trim OUT timestamps. Base these cuts on specific visual action cues (e.g., "Cut exactly as her shoulder drops," or "Start right as the glass touches his lips"). Do not just give numbers; explain the visual cue.
4. Transitions: Default to Hard Cuts ("None"). Short-form social media ads perform best with snappy hard cuts. If you recommend a transition (like a Dissolve), you must justify exactly why it is narratively necessary. OUTPUT FORMAT: Always present your final editing blueprint in the following format: Overall Strategy: (Briefly explain the pacing, what clips you decided to skip and why, and the total estimated length of the flexible ad). Clip 1: [Name of Scene/Action] Raw File Used: [Identify the file based on the file name and the visual description, e.g., "The wide shot in the gym", sometimes there could be different versions of the same clip]. Trim IN: 00:0X (Describe the visual starting point) Trim OUT: 00:0X (Describe the visual ending point) Transition to next clip: [Hard Cut / Dissolve] Why this cut: (Explain the narrative or pacing reason). (Repeat for all necessary clips) Loop/Ending Check: (Confirm exactly how the final frame connects back to the first frame visually, and advise on any audio cues needed for the loop)."""

EDL_JSON_RIDER = """Return your blueprint ALSO as a single JSON object matching the provided schema. For EVERY clip you keep, echo the exact clip_label, s3_key, and version from the manifest — never paraphrase the file, and never use two versions of the same shot_id. For every shot marked choose_one, watch its candidate versions, pick exactly ONE, and record it in version_choices (chosen_s3_key, chosen_version, considered, reason). All timestamps in seconds as floats. transition_to_next is 'none' unless a dissolve is strictly justified; set transition_duration_sec accordingly (0 for hard cuts)."""


class SkipNote(BaseModel):
    clip_label: str
    s3_key: str
    reason: str


class VersionChoice(BaseModel):
    shot_id: str
    chosen_s3_key: str
    chosen_version: str
    considered: List[str]
    reason: str


class LoopCheck(BaseModel):
    is_seamless_loop_possible: bool
    reasoning: str
    recommendation: str


class EDLClip(BaseModel):
    order: int
    scene_action_name: str
    clip_label: str
    s3_key: str
    version: str
    trim_in_sec: float
    trim_in_cue: str
    trim_out_sec: float
    trim_out_cue: str
    transition_to_next: str
    transition_duration_sec: float
    why_this_cut: str


class EditDecisionList(BaseModel):
    overall_strategy: str
    estimated_length_sec: float
    clips: List[EDLClip]
    loop_check: LoopCheck
    version_choices: List[VersionChoice]
    skipped_clips: List[SkipNote]


class EDLGeneratorAgent:
    def __init__(self):
        self.tmp_dir = os.environ.get("PHASE4_TMP_DIR", "/tmp/phase4")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.pipeline_service = PipelineService()
        
    def _mint_presigned_url(self, s3_client, bucket: str, s3_key: str) -> str:
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=86400 * 7,
        )

    def process(self, state: Phase4State) -> Phase4State:
        show_id = state.get("show_id")
        episode_number = state.get("episode_number")
        job_id = state.get("job_id")
        clip_manifest = state.get("clip_manifest", [])
        
        logger.info(f"Agent 1 EDL Generator starting for show_id={show_id}, ep={episode_number}, job_id={job_id}")

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=1, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status: {e}")

        mongo_client = None
        s3_client = None
        genai_client = None
        uploaded_files = []
        local_clip_paths = []

        try:
            mongo_uri = os.environ.get("MONGODB_ATLAS_URI")
            if mongo_uri:
                mongo_client = MongoClient(mongo_uri)
                db = mongo_client.get_database("production")  # Update with appropriate default later if needed
            else:
                db = None
                
            bucket_name = os.environ.get("production_S3_BUCKET_NAME", "zeroshot-v1")
            
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.environ.get("production_AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("production_AWS_SECRET_ACCESS_KEY"),
                region_name=os.environ.get("production_AWS_REGION", "eu-north-1"),
            )
            
            genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

            # 1. Prepare candidates and upload to Gemini
            flattened_candidates = []
            grouped_text_manifest = []
            candidate_map = {}  # s3_key -> (Duration, shot_id)

            idx = 0
            for group in clip_manifest:
                shot_id = group.get("shot_id")
                selection_mode = group.get("selection_mode")
                
                group_text = []
                group_text.append(f"Shot ID: {shot_id} | Selection Mode: {selection_mode}")
                if selection_mode == "single":
                    group_text.append("  (Use the candidate provided exactly - NO choices allowed)")
                elif selection_mode == "choose_one":
                    group_text.append("  (You MUST pick exactly ONE candidate from these variations for the final ad. Never use multiple variations of the same shot.)")
                
                for candidate in group.get("candidates", []):
                    idx += 1
                    clip_label = f"CLIP_{chr(64 + idx)}" if idx <= 26 else f"CLIP_{idx}"
                    s3_key = candidate.get("s3_key")
                    version = candidate.get("version")
                    duration = candidate.get("duration", 0.0)
                    
                    candidate_map[s3_key] = {"duration": duration, "shot_id": shot_id}
                    
                    # Regenerate Presigned URL and Download
                    signed_url = self._mint_presigned_url(s3_client, bucket_name, s3_key)
                    local_path = os.path.join(self.tmp_dir, f"{str(uuid.uuid4())[:8]}_{os.path.basename(s3_key)}")
                    local_clip_paths.append(local_path)

                    with requests.get(signed_url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(local_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                                
                    # Upload to Gemini Files API
                    uploaded_file = genai_client.files.upload(file=local_path)
                    uploaded_file = wait_for_gemini_file_active(genai_client, uploaded_file)
                    uploaded_files.append(uploaded_file)
                    
                    flattened_candidates.append({
                        "label": clip_label,
                        "s3_key": s3_key,
                        "version": version,
                        "shot_id": shot_id,
                        "file_handle": uploaded_file
                    })
                    
                    group_text.append(f"  - Label: {clip_label} | s3_key: {s3_key} | version: {version} | duration: {duration}s | File Name: {candidate.get('filename')} | Description: {candidate.get('description', '')}")
                    
                grouped_text_manifest.append("\n".join(group_text))
            
            manifest_text = "\n\n".join(grouped_text_manifest)

            # 2. Build Contents for model
            contents = [
                DR_EDITOR_SYSTEM_PROMPT + "\n\n" + EDL_JSON_RIDER,
                f"Script:\n{state.get('script_content', '')}",
                f"Shot List (guideline only):\n{json.dumps(state.get('shot_list', {}), indent=2)}",
                f"Grouped Text Manifest:\n{manifest_text}"
            ]
            
            for c in flattened_candidates:
                contents.append(f"=== {c['label']} (shot_id={c['shot_id']}, s3_key={c['s3_key']}, version={c['version']}) ===")
                contents.append(c["file_handle"])
                
            # 3. Call Model with Retries
            max_retries = 3
            backoff = [2, 4, 8]
            response = None
            for attempt in range(max_retries):
                try:
                    response = genai_client.models.generate_content(
                        model="gemini-3.1-pro-preview",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=EditDecisionList
                        )
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Gemini API attempt {attempt+1} failed: {e}. Retrying.")
                        time.sleep(backoff[attempt])
                    else:
                        raise e

            if not response or not response.parsed:
                raise ValueError("Model returned empty or unparseable response.")

            parsed_edl: EditDecisionList = response.parsed
            
            # 4. Validation
            valid_clips = []
            used_shot_ids = set()
            for clip in parsed_edl.clips:
                if clip.s3_key not in candidate_map:
                    logger.warning(f"Model invented s3_key '{clip.s3_key}'. Dropping clip.")
                    state.setdefault("errors", []).append({"agent": "agent1", "error": f"Invalid s3_key {clip.s3_key} dropped."})
                    continue
                    
                clip_data = candidate_map[clip.s3_key]
                if clip_data["shot_id"] in used_shot_ids:
                    logger.warning(f"Multiple versions used for shot {clip_data['shot_id']}. Dropping this extra one.")
                    state.setdefault("errors", []).append({"agent": "agent1", "error": f"Extra version for shot {clip_data['shot_id']} dropped."})
                    continue
                    
                used_shot_ids.add(clip_data["shot_id"])
                
                # Clamp trims
                dur = clip_data["duration"]
                clip.trim_in_sec = max(0.0, min(clip.trim_in_sec, dur))
                clip.trim_out_sec = max(0.0, min(clip.trim_out_sec, dur))
                if clip.trim_out_sec <= clip.trim_in_sec:
                    clip.trim_out_sec = min(clip.trim_in_sec + 0.4, dur)
                if clip.trim_out_sec - clip.trim_in_sec < 0.4:
                    clip.trim_out_sec = min(clip.trim_in_sec + 0.4, dur)
                
                valid_clips.append(clip)
                
            valid_clips.sort(key=lambda x: x.order)
            for i, c in enumerate(valid_clips):
                c.order = i + 1
            parsed_edl.clips = valid_clips
            
            # Check choose_one constraint
            for group in clip_manifest:
                if group.get("selection_mode") == "choose_one":
                    shot_id = group.get("shot_id")
                    choice_record = next((vc for vc in parsed_edl.version_choices if vc.shot_id == shot_id), None)
                    if not choice_record:
                        # Find if any clip belongs to this
                        used = next((c for c in valid_clips if candidate_map.get(c.s3_key, {}).get("shot_id") == shot_id), None)
                        if used:
                            parsed_edl.version_choices.append(VersionChoice(
                                shot_id=shot_id,
                                chosen_s3_key=used.s3_key,
                                chosen_version=used.version,
                                considered=[c["s3_key"] for c in group.get("candidates", [])],
                                reason="Fallback recorded by system based on used clip."
                            ))
                            
            edl_dict = parsed_edl.model_dump()
            
            # 5. Store
            state["edl"] = edl_dict
            state["edl_version"] = 0
            state["current_agent"] = "agent1"
            
            if db is not None:
                try:
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=1,
                        status="completed",
                        output={"edl": edl_dict, "edl_version": 0}
                    )
                    append_versioned(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        array_field="edl_versions",
                        entry={
                            "version": 0,
                            "edl": edl_dict,
                            "created_at": datetime.now(timezone.utc).isoformat()
                        }
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 1: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=1, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="agent_2")
                except Exception as e:
                    pass
                    
            logger.info("Agent 1 EDL Generator completed successfully.")
            return state

        except Exception as e:
            logger.error(f"Agent 1 EDL Generator failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent1", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=1, status="failed")
                except Exception:
                    pass
            raise e

        finally:
            # Cleanup Gemini Files API uploads
            if genai_client and uploaded_files:
                for f in uploaded_files:
                    try:
                        genai_client.files.delete(name=f.name)
                    except Exception as e:
                        logger.warning(f"Could not delete file {f.name} from Gemini API: {e}")
            # Cleanup downloaded clip temp files
            for p in local_clip_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {p}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_1_edl_node(state: Phase4State) -> Phase4State:
    agent = EDLGeneratorAgent()
    return agent.process(state)
