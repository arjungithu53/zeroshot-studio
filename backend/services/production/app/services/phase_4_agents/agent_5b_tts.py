import io
import os
import re
import time
import wave
import logging
import subprocess
from datetime import datetime, timezone
from typing import Optional

import boto3
from pymongo import MongoClient
from google import genai
from google.genai import types

from app.services.final_assemblies_service import update_agent_output, append_versioned
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.workflow_state import Phase4State

logger = logging.getLogger(__name__)

__all__ = ["TTSAgent", "agent_5b_tts_node"]

# PCM spec returned by Gemini Flash TTS
_PCM_RATE = 24_000
_PCM_CHANNELS = 1
_PCM_SAMPWIDTH = 2  # 16-bit


def _ensure_wav(audio_bytes: bytes) -> bytes:
    """Return WAV bytes — wrap PCM if there is no RIFF header yet."""
    if audio_bytes[:4] == b"RIFF":
        return audio_bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_PCM_CHANNELS)
        wf.setsampwidth(_PCM_SAMPWIDTH)
        wf.setframerate(_PCM_RATE)
        wf.writeframes(audio_bytes)
    return buf.getvalue()


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
        return 0.0


def _atempo_filter(speed: float) -> str:
    """Build an FFmpeg atempo chain for tempo factors outside the single-filter range."""
    factors = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6g}" for factor in factors)


def _fit_audio_duration(path: str, target_duration: float) -> tuple[str, float, float]:
    """Speed-adjust a WAV if it exceeds the target duration."""
    current_duration = _ffprobe_duration(path)
    if current_duration <= 0 or target_duration <= 0:
        return path, current_duration, 1.0
    if current_duration <= target_duration:
        return path, current_duration, 1.0

    speed = current_duration / target_duration
    out_path = path.replace(".wav", "_fit.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", path,
        "-filter:a", _atempo_filter(speed),
        "-ar", str(_PCM_RATE),
        "-ac", str(_PCM_CHANNELS),
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"VO duration fit failed: {result.stderr}")

    fitted_duration = _ffprobe_duration(out_path)
    logger.info(
        f"VO duration fit: {current_duration:.2f}s -> {fitted_duration:.2f}s "
        f"(target={target_duration:.2f}s, speed={speed:.2f}x)"
    )
    return out_path, fitted_duration, speed


def _assemble_tts_prompt(plan: dict, title: str) -> str:
    """
    Build the advanced TTS prompt in the exact structure recommended by the Gemini docs.
    Leads with a preamble to prevent the model reading headers/notes aloud
    (documented 'prompt classifier false rejection' issue).
    """
    persona = plan.get("persona", {})
    scene = plan.get("scene", {})
    director_notes = plan.get("director_notes", {})

    lines = [
        "Synthesize speech for the following performance. "
        "Read ONLY the TRANSCRIPT section — "
        "do not read the headers, labels, director notes, or any bracketed instructions aloud.",
        "",
        f"# AUDIO PROFILE: {persona.get('name', 'Narrator')}",
        f'## "{title}"',
        "",
        f"## THE SCENE: {scene.get('location', '')}",
        scene.get("environment_details", ""),
        "",
        "### DIRECTOR'S NOTES",
        f"Style: {director_notes.get('style', '')}",
        f"Pacing: {director_notes.get('pacing', '')}",
        f"Accent: {director_notes.get('accent', '')}",
        "",
        "### SAMPLE CONTEXT",
        plan.get("sample_context", ""),
        "",
        "#### TRANSCRIPT",
        plan.get("transcript_with_tags", ""),
    ]
    return "\n".join(lines)


class TTSAgent:

    def __init__(self):
        self.tmp_dir = os.environ.get("PHASE4_TMP_DIR", "/tmp/phase4")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.pipeline_service = PipelineService()

    def _call_with_retry(self, fn, *args, **kwargs):
        backoff = [3, 6, 12]
        for attempt in range(3):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"TTS attempt {attempt + 1} failed: {e}. Retrying in {backoff[attempt]}s.")
                    time.sleep(backoff[attempt])
                else:
                    raise

    def _mint_presigned_url(self, s3_client, bucket: str, s3_key: str) -> str:
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=86400 * 7,
        )

    def process(self, state: Phase4State) -> Phase4State:
        show_id = state.get("show_id")
        episode_id = state.get("episode_id")
        episode_number = state.get("episode_number")
        job_id = state.get("job_id")
        title = state.get("title", "untitled")
        vo_director_plan = state.get("vo_director_plan")
        assembled_duration = float(state.get("assembled_duration") or 0.0)

        logger.info(
            f"Agent 5B TTS starting: show_id={show_id}, ep={episode_number}, "
            f"voice={vo_director_plan.get('persona', {}).get('recommended_voice', '?') if vo_director_plan else 'no plan'}"
        )

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=5, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status: {e}")

        if not vo_director_plan:
            raise ValueError("vo_director_plan is missing from state — Agent 5A must run first.")

        mongo_client = None
        s3_client = None
        local_wav_path = None

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

            # 1. Choose voice — env override takes precedence over director's pick.
            # Use .strip() + "or" so an empty PHASE4_TTS_VOICE="" falls back to the
            # director's recommendation instead of passing an empty string to the API.
            recommended_voice = vo_director_plan.get("persona", {}).get("recommended_voice", "Kore")
            tts_voice = os.environ.get("PHASE4_TTS_VOICE", "").strip() or recommended_voice
            logger.info(f"TTS voice selected: {tts_voice} (recommended={recommended_voice})")

            # 2. Assemble prompt
            assembled_prompt = _assemble_tts_prompt(vo_director_plan, title)
            state["vo_tts_prompt"] = assembled_prompt
            logger.debug(f"Assembled TTS prompt ({len(assembled_prompt)} chars).")

            # 3. Call Gemini Flash TTS
            logger.info("Calling gemini-3.1-flash-tts-preview.")
            response = self._call_with_retry(
                genai_client.models.generate_content,
                model="gemini-3.1-flash-tts-preview",
                contents=assembled_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=tts_voice,
                            )
                        )
                    ),
                ),
            )

            if (
                not response
                or not response.candidates
                or not response.candidates[0].content.parts
            ):
                raise ValueError("TTS API returned an empty response.")

            # Collect all audio parts — long VO may be returned as multiple PCM chunks.
            # Concatenating raw PCM bytes before wrapping in a single WAV header is correct.
            audio_chunks = [
                p.inline_data.data
                for p in response.candidates[0].content.parts
                if hasattr(p, "inline_data") and p.inline_data and p.inline_data.data
            ]
            if not audio_chunks:
                first_part = response.candidates[0].content.parts[0]
                raise ValueError(
                    f"TTS API returned text instead of audio — possible classifier rejection. "
                    f"Part text: {getattr(first_part, 'text', '')[:200]}"
                )

            raw_audio = b"".join(audio_chunks)

            # 4. Ensure WAV wrapper
            wav_bytes = _ensure_wav(raw_audio)

            # 5. Write to temp file and probe duration
            vo_version = state.get("vo_version", 0) + 1
            title_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", title)[:50]
            wav_filename = f"{title_safe}_VO_v{vo_version}.wav"
            local_wav_path = os.path.join(self.tmp_dir, wav_filename)
            with open(local_wav_path, "wb") as f:
                f.write(wav_bytes)

            vo_duration = _ffprobe_duration(local_wav_path)
            logger.info(f"VO WAV written: {wav_filename} ({vo_duration:.2f}s, {len(wav_bytes)} bytes)")

            duration_fit_speed = 1.0
            if assembled_duration > 0:
                target_vo_duration = assembled_duration * float(os.environ.get("PHASE4_VO_TARGET_RATIO", "0.96"))
                fitted_path, fitted_duration, duration_fit_speed = _fit_audio_duration(
                    local_wav_path,
                    target_vo_duration,
                )
                if fitted_path != local_wav_path:
                    local_wav_path = fitted_path
                    vo_duration = fitted_duration
                    wav_filename = os.path.basename(local_wav_path)

            # 6. Upload to S3
            ep_id = episode_id or f"ep{episode_number}"
            s3_key = f"phase4/{show_id}/{ep_id}/audio/{wav_filename}"
            s3_client.upload_file(
                local_wav_path,
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "audio/wav"},
            )
            signed_url = self._mint_presigned_url(s3_client, bucket_name, s3_key)
            logger.info(f"VO uploaded to S3: {s3_key}")

            # 7. Update state
            state["vo_s3_key"] = s3_key
            state["vo_s3_url"] = signed_url
            state["vo_version"] = vo_version
            state["vo_duration"] = vo_duration
            state["current_agent"] = "agent5b"

            # 8. Persist to MongoDB
            if db is not None:
                try:
                    vo_entry = {
                        "version": vo_version,
                        "s3_key": s3_key,
                        "s3_url": signed_url,
                        "duration": vo_duration,
                        "duration_fit_speed": duration_fit_speed,
                        "assembled_duration": assembled_duration,
                        "voice": tts_voice,
                        "director_plan": vo_director_plan,
                        "tts_prompt_length": len(assembled_prompt),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=5,
                        status="completed",
                        output={"stage": "tts", **vo_entry},
                    )
                    append_versioned(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        array_field="vo",
                        entry=vo_entry,
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 5B: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=5, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="agent_6")
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after 5B: {e}")

            logger.info("Agent 5B TTS completed successfully.")
            return state

        except Exception as e:
            logger.error(f"Agent 5B TTS failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent5b", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=5, status="failed")
                except Exception:
                    pass
            raise

        finally:
            if local_wav_path and os.path.exists(local_wav_path):
                try:
                    os.remove(local_wav_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {local_wav_path}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_5b_tts_node(state: Phase4State) -> Phase4State:
    agent = TTSAgent()
    return agent.process(state)
