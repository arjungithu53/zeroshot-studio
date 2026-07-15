import os
import re
import time
import logging
import subprocess
from datetime import datetime, timezone

import boto3
from pymongo import MongoClient
from google import genai

from app.services.final_assemblies_service import update_agent_output, append_versioned
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.workflow_state import Phase4State

logger = logging.getLogger(__name__)

__all__ = ["LyriaAgent", "agent_7b_lyria_node"]

_LYRIA_CLIP_DURATION = 30.0  # Lyria 3 Clip always generates exactly 30 seconds


def _ffprobe_duration(path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"ffprobe failed for {path}: {e}")
        return _LYRIA_CLIP_DURATION


def _convert_and_trim(mp3_path: str, wav_path: str, trim_to: float | None) -> None:
    """Convert MP3 → WAV, optionally trimming to trim_to seconds."""
    cmd = ["ffmpeg", "-y", "-i", mp3_path]
    if trim_to is not None:
        cmd += ["-t", str(trim_to)]
    cmd += ["-ar", "44100", "-ac", "2", wav_path]
    logger.info(f"FFmpeg convert/trim: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed (code {result.returncode}): {result.stderr}")


class LyriaAgent:

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

    def _call_with_retry(self, fn, *args, **kwargs):
        backoff = [3, 6, 12]
        for attempt in range(3):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Lyria attempt {attempt + 1} failed: {e}. Retrying in {backoff[attempt]}s.")
                    time.sleep(backoff[attempt])
                else:
                    raise

    def process(self, state: Phase4State) -> Phase4State:
        show_id = state.get("show_id")
        episode_id = state.get("episode_id")
        episode_number = state.get("episode_number")
        job_id = state.get("job_id")
        title = state.get("title", "untitled")
        music_director_plan = state.get("music_director_plan")
        assembled_duration = state.get("assembled_duration", 0.0)

        if not music_director_plan:
            raise ValueError("music_director_plan is missing from state — Agent 7A must run first.")

        filled_prompt = music_director_plan.get("filled_prompt", "")
        volume_envelope = music_director_plan.get("volume_envelope", [])
        timing_map = music_director_plan.get("timing_map", [])

        logger.info(
            f"Agent 7B Lyria starting: show_id={show_id}, ep={episode_number}, "
            f"prompt_length={len(filled_prompt)} chars"
        )

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=7, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status: {e}")

        mongo_client = None
        local_mp3 = None
        local_wav = None

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

            lyria_model = os.environ.get("PHASE4_LYRIA_MODEL", "lyria-3-clip-preview")

            # 1. Call Lyria
            logger.info(f"Calling {lyria_model} for music generation.")
            response = self._call_with_retry(
                genai_client.models.generate_content,
                model=lyria_model,
                contents=filled_prompt,
            )

            # 2. Parse response — iterate parts, find inline_data (MP3 bytes)
            candidates = response.candidates or []
            if not candidates:
                block_reason = getattr(response.prompt_feedback, "block_reason", None)
                raise ValueError(f"Lyria returned no candidates (prompt_feedback block_reason={block_reason}).")

            finish_reason = getattr(candidates[0], "finish_reason", None)
            parts = response.parts
            if not parts:
                raise ValueError(
                    f"Lyria returned no content parts (finish_reason={finish_reason}, "
                    f"prompt_feedback={response.prompt_feedback})."
                )

            mp3_bytes = None
            for part in parts:
                if part.text is not None:
                    logger.info(f"Lyria text part (lyrics/structure): {part.text[:300]}")
                elif part.inline_data is not None:
                    mp3_bytes = part.inline_data.data
                    logger.info(f"Lyria audio part: {len(mp3_bytes)} bytes")

            if not mp3_bytes:
                raise ValueError("Lyria returned no audio data — all parts were text.")

            # 3. Write MP3 to temp
            music_version = state.get("music_version", 0) + 1
            title_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", title)[:50]
            local_mp3 = os.path.join(self.tmp_dir, f"7b_music_{title_safe}_v{music_version}.mp3")
            local_wav = os.path.join(self.tmp_dir, f"7b_music_{title_safe}_v{music_version}.wav")

            with open(local_mp3, "wb") as f:
                f.write(mp3_bytes)
            logger.info(f"MP3 written: {local_mp3} ({len(mp3_bytes)} bytes)")

            # 4. Duration handling — Lyria Clip is always ~30s
            # Trim to ad duration if ad is shorter; if longer, use as-is (silence after 30s)
            trim_to = assembled_duration if assembled_duration < _LYRIA_CLIP_DURATION else None
            if trim_to is not None:
                logger.info(f"Ad ({assembled_duration:.2f}s) < 30s — trimming music to {trim_to:.2f}s.")
            else:
                logger.info(f"Ad ({assembled_duration:.2f}s) >= 30s — keeping full 30s clip.")

            # 5. Convert MP3 → WAV (and optionally trim)
            _convert_and_trim(local_mp3, local_wav, trim_to)
            music_duration = _ffprobe_duration(local_wav)
            logger.info(f"WAV ready: {local_wav} ({music_duration:.2f}s)")

            # 6. Upload to S3
            wav_filename = f"{title_safe}_MUSIC_v{music_version}.wav"
            ep_id = episode_id or f"ep{episode_number}"
            s3_key = f"phase4/{show_id}/{ep_id}/audio/{wav_filename}"
            s3_client.upload_file(
                local_wav,
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "audio/wav"},
            )
            signed_url = self._mint_presigned_url(s3_client, bucket_name, s3_key)
            logger.info(f"Music uploaded to S3: {s3_key}")

            # 7. Update state
            state["music_s3_key"] = s3_key
            state["music_s3_url"] = signed_url
            state["music_version"] = music_version
            state["music_duration"] = music_duration
            state["current_agent"] = "agent7b"

            # 8. Persist to MongoDB
            if db is not None:
                try:
                    music_entry = {
                        "version": music_version,
                        "s3_key": s3_key,
                        "s3_url": signed_url,
                        "duration": music_duration,
                        "prompt": filled_prompt,
                        "timing_map": timing_map,
                        "volume_envelope": volume_envelope,
                        "lyria_model": lyria_model,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=7,
                        status="completed",
                        output={"stage": "lyria", **music_entry},
                    )
                    append_versioned(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        array_field="music",
                        entry=music_entry,
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 7B: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=7, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="agent_8")
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after 7B: {e}")

            logger.info("Agent 7B Lyria completed successfully.")
            return state

        except Exception as e:
            logger.error(f"Agent 7B Lyria failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent7b", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=7, status="failed")
                except Exception:
                    pass
            raise

        finally:
            for path in (local_mp3, local_wav):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {path}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_7b_lyria_node(state: Phase4State) -> Phase4State:
    agent = LyriaAgent()
    return agent.process(state)
