import os
import time
import logging
import requests
from datetime import datetime, timezone
from typing import List

import boto3
from pymongo import MongoClient
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.services.final_assemblies_service import update_agent_output
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.gemini_files import wait_for_gemini_file_active
from app.services.phase_4_agents.workflow_state import Phase4State

logger = logging.getLogger(__name__)

__all__ = ["MusicDirectorAgent", "agent_7a_music_director_node"]

MUSIC_DIRECTOR_SYSTEM_PROMPT = """You are an expert Music Supervisor and Audio Director for short-form Direct-Response ads. You will watch the FINAL edited video (which already contains the voiceover) and read the script, then produce a single, finished text prompt for an instrumental music-generation model (Lyria) by filling the provided template.

RULES:
1. The video is the source of truth. Set the track DURATION to match the final video's exact length (do not default to 30 seconds unless the video is 30 seconds). Place every transition / beat-drop timestamp on a REAL visual change you observe (product reveal, scene change, the emotional turn), expressed in seconds from 0:00.
2. Serve the voiceover, not compete with it. Choose a genre/vibe, energy, and instrumentation that sits UNDER spoken narration — avoid busy midrange and lead melodies that fight the voice. The music will be mixed against the VO downstream, so design it to breathe.
3. Strictly instrumental. No vocals, no vocal samples, no voice-like leads. Always include "Instrumental only, no vocals" explicitly in the filled prompt.
4. Match the brand and emotional arc you see on screen. Open to reflect the opening visual/mood, lift at the key reveal, and resolve on the ending beat.
5. Fill EVERY placeholder in the template with concrete choices (adjective, genre/vibe, video type, 3–5 specific instruments/textures, and the exact start/transition/climax descriptions and timestamps). Output ONLY the finished prompt text plus the JSON structured data.
6. Volume Envelope: Output a `volume_envelope` — a list of {start_sec, end_sec, volume} entries that together cover the FULL track duration with no gaps. `volume` is a mix weight (0.0–1.0) that Agent 8 will apply to the music stem at that moment relative to the voiceover. Rules: set lower values (0.10–0.20) under dense narration sections, raise to 0.40–0.60 at product reveals or emotional peaks, and never leave gaps between sections."""

LYRIA_TEMPLATE = """Create a 30-second [Adjective] [Genre/Vibe] instrumental track for a [Type of Video]. Instruments: [List 3-5 specific instruments, textures, or SFX qualities]. Vocals: Strictly instrumental, absolutely no vocals or voice-like sounds. Structure:

* Start (0:00): Begin with [describe the opening sound/mood] to reflect [opening visual].
* Transition (0:0X): At the [X]-second mark, introduce [new sound/instrument/beat drop] to match a sudden [describe visual change, e.g., product reveal, scene change].
* Climax/Ending (0:0X to End): Build toward a [describe final mood] climax using [specific instrument] to signify [final emotion/conclusion], holding this energy until the end."""


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class TimingEvent(BaseModel):
    at_sec: float
    event: str
    musical_action: str


class VolumeSection(BaseModel):
    start_sec: float
    end_sec: float
    volume: float


class MusicDirectorPlan(BaseModel):
    filled_prompt: str
    duration_sec: float
    timing_map: List[TimingEvent]
    volume_envelope: List[VolumeSection]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MusicDirectorAgent:

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

    def _download(self, url: str, local_path: str) -> None:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def _call_with_retry(self, fn, *args, **kwargs):
        backoff = [2, 4, 8]
        for attempt in range(3):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Gemini attempt {attempt + 1} failed: {e}. Retrying in {backoff[attempt]}s.")
                    time.sleep(backoff[attempt])
                else:
                    raise

    def process(self, state: Phase4State) -> Phase4State:
        show_id = state.get("show_id")
        episode_number = state.get("episode_number")
        job_id = state.get("job_id")
        vo_preview_s3_key = state.get("vo_preview_s3_key")
        script_content = state.get("script_content", "")
        assembled_duration = state.get("assembled_duration", 0.0)

        logger.info(
            f"Agent 7A Music Director starting: show_id={show_id}, ep={episode_number}, "
            f"assembled_duration={assembled_duration:.2f}s"
        )

        if not vo_preview_s3_key:
            raise ValueError("vo_preview_s3_key is missing from state — Agent 6 must run first.")

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=7, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status: {e}")

        mongo_client = None
        genai_client = None
        uploaded_file = None
        local_path = None

        try:
            mongo_uri = os.environ.get("MONGODB_ATLAS_URI")
            if mongo_uri:
                mongo_client = MongoClient(mongo_uri)
                db = mongo_client.get_database("production")
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

            # 1. Download VO preview and upload to Gemini Files API
            signed_url = self._mint_presigned_url(s3_client, bucket_name, vo_preview_s3_key)
            local_path = os.path.join(self.tmp_dir, f"7a_preview_{os.path.basename(vo_preview_s3_key)}")
            logger.info(f"Downloading VO preview: {vo_preview_s3_key}")
            self._download(signed_url, local_path)

            logger.info("Uploading VO preview to Gemini Files API.")
            uploaded_file = genai_client.files.upload(file=local_path)
            uploaded_file = wait_for_gemini_file_active(genai_client, uploaded_file)

            # 2. Build contents
            contents = [
                MUSIC_DIRECTOR_SYSTEM_PROMPT,
                f"LYRIA TEMPLATE TO FILL (replace every bracketed placeholder with concrete choices):\n{LYRIA_TEMPLATE}",
                f"SCRIPT (for tonal/thematic reference):\n{script_content}",
                f"TOTAL AD DURATION: {assembled_duration:.2f} seconds. "
                f"Note: Lyria 3 Clip always generates exactly 30 seconds. "
                f"If the ad is shorter than 30s, fill in duration_sec with the actual ad length so downstream trimming is accurate.",
                "VO PREVIEW VIDEO (with narration — score AROUND the voice):",
                uploaded_file,
            ]

            # 3. Director call
            logger.info("Calling gemini-3.1-pro-preview for music direction.")
            response = self._call_with_retry(
                genai_client.models.generate_content,
                model="gemini-3.1-pro-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MusicDirectorPlan,
                ),
            )

            if not response or not response.parsed:
                raise ValueError("Music Director call returned an empty or unparseable response.")

            plan: MusicDirectorPlan = response.parsed
            plan_dict = plan.model_dump()

            # Validate volume envelope covers the full duration with no large gaps
            envelope = sorted(plan.volume_envelope, key=lambda s: s.start_sec)
            if envelope:
                coverage_end = envelope[-1].end_sec
                if coverage_end < assembled_duration * 0.9:
                    logger.warning(
                        f"Volume envelope ends at {coverage_end:.1f}s but ad is {assembled_duration:.1f}s. "
                        f"Adding a fallback section."
                    )
                    plan_dict["volume_envelope"].append({
                        "start_sec": coverage_end,
                        "end_sec": assembled_duration,
                        "volume": 0.15,
                    })

            logger.info(
                f"7A Music Director complete. Prompt length={len(plan.filled_prompt)} chars, "
                f"timing events={len(plan.timing_map)}, "
                f"volume sections={len(plan.volume_envelope)}"
            )

            # 4. Update state
            state["music_director_plan"] = plan_dict
            state["music_prompt"] = plan.filled_prompt
            state["music_plan"] = [t.model_dump() for t in plan.timing_map]
            state["current_agent"] = "agent7a"

            # 5. Persist to MongoDB
            if db is not None:
                try:
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=7,
                        status="completed",
                        output={"stage": "director", "music_director_plan": plan_dict},
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 7A: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=7, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="agent_7b")
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after 7A: {e}")

            logger.info("Agent 7A Music Director completed successfully.")
            return state

        except Exception as e:
            logger.error(f"Agent 7A Music Director failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent7a", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=7, status="failed")
                except Exception:
                    pass
            raise

        finally:
            if genai_client and uploaded_file:
                try:
                    genai_client.files.delete(name=uploaded_file.name)
                except Exception as e:
                    logger.warning(f"Could not delete Gemini file {uploaded_file.name}: {e}")
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {local_path}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_7a_music_director_node(state: Phase4State) -> Phase4State:
    agent = MusicDirectorAgent()
    return agent.process(state)
