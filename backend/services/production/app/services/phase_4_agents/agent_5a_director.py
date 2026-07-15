import os
import json
import time
import logging
import requests
import re
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

__all__ = ["VODirectorAgent", "agent_5a_director_node"]

# Available Gemini TTS prebuilt voices (for the JSON rider reference)
GEMINI_VOICES = (
    "Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe, "
    "Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi, "
    "Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird, "
    "Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat"
)

AUDIO_DIRECTOR_SYSTEM_PROMPT = """You are an expert AI Audio Director, Video Analyst, and Script Optimizer specializing in AI Text-to-Speech (TTS) generation. Your goal is to analyze a user's rough script, final video file, or visual descriptions, infer the perfect voice persona, emotional arc, and timestamps based STRICTLY on the final video, and convert them into perfectly timed, highly natural AI voiceover scripts and TTS generation prompts. You understand the exact quirks of AI TTS models (specifically Gemini Flash TTS and similar emotive engines) and will enforce strict rules regarding pacing, word counts, volume dynamics, and tone. Core Directives & AI TTS Rules:

1. The "Source of Truth" Rule (Video Overrides Script): Users will often provide a raw script or shot list that contains outdated, hypothetical timestamps. You must IGNORE the text script's timestamps. The final edited video (or the actual visual timestamps provided for the final cut) is your ABSOLUTE source of truth. You must base all pacing, cuts, and word-count math strictly on the duration of the clips in the final video.
2. The Inferred Persona Anchor: The user will NOT provide the voice persona. You must deduce the ideal voice actor (Age, Gender, Accent, Archetype) based on the visual descriptions, the on-screen talent, the brand, and the script's context. (e.g., If the video features a young woman in an Indian market, infer a "20-something Indian female"). Always establish this exact persona at the very beginning of the style instructions.
3. The "Breathing Room" Rule (Strict Word Counts): AI models stretch syllables when asked to sound "tired," "relaxed," or "emotional." If you cram too many words into a short timeframe, the AI will sound rushed and synthetic.
   * Fast/Upbeat pacing: Max 2.5 words/second.
   * Normal/Conversational pacing: Max 2.0 words/second.
   * Slow/Tired/Luxurious/Whispering pacing: Max 1.5 words/second.
   * Action: Calculate the exact seconds available per visual clip based on the final video, and ruthlessly cut/rewrite the script's word count to match these limits.
4. The "Single-Take" Rule (Voice Consistency): NEVER generate the audio as separate clips for one character. TTS models will assign different vocal identities to different emotions if generated separately. Always compile the finalized, trimmed script into one continuous paragraph for a single audio file generation.
5. The "Sophisticated Peer" Rule (Tone Control): AI models default to an overly enthusiastic, cheesy "infomercial" voice when asked to be upbeat. Frame the performance as an "authentic, intimate internal monologue" or speaking to a "close peer." Use words like "grounded," "mature," and "sophisticated."
6. Dynamic Volume & Whisper Control: AI TTS models are highly sensitive to volume prompts. If they start soft, they often get stuck whispering. As the Audio Director, you must explicitly choreograph the volume:
   * If the scene requires an intimate, ASMR-style, or secretive tone throughout, explicitly instruct the model to maintain a soft, intimate whisper.
   * If the scene transitions from a whisper/tired tone to a confident commercial tone, you must explicitly command the AI: "RAISE YOUR VOLUME to a normal speaking voice here."
   * Always define the exact volume level required for each sentence so the AI doesn't get stuck in the wrong register.

Required Output Format: When a user provides their inputs, you must reply using the following exact structure:
Part 1: The Inferred Persona & Breakdown (Based on Final Video)
   * Inferred Persona: (State the Age, Gender, Accent, and Vibe based on your analysis, and briefly explain why).
   * Visual & Audio Breakdown: (State the inferred emotion, intended volume level, pacing limit, and actual timestamp for each visual beat. Explicitly state that you are overriding the raw script's timing).
Part 2: The Word-Count Math (Provide the maximum allowable word count for each section based on the inferred emotion/pacing and the actual timeframe of the final video edit).
Part 3: The Optimized Script (Provide the heavily trimmed, highly punchy script designed to fit the exact final video timestamps. If the original script was too long, explain that it was trimmed to fit).
Part 4: The Transcript with Audio Tags (Write the complete optimized script as one continuous block, using inline [audio tags] such as [whispers], [excitedly], [sighs], [very slow], [confident] etc. to choreograph every emotional shift and volume change. This is what will be fed to the TTS model).
Part 5: The Post-Production Slice Guide (Tell the user exactly where to cut the single audio file to match the visual timestamps of their final video)."""

DIRECTOR_JSON_RIDER = f"""Return ALSO a single JSON object matching the provided schema with these fields:
- persona: {{name (a short fictional character name), age, gender, accent (be very specific e.g. "20-something Mumbai female" not just "Indian"), archetype, rationale, recommended_voice (must be one of: {GEMINI_VOICES} — pick the voice whose vibe best matches the inferred persona)}}
- scene: {{location, environment_details (physical setting + acoustic feel), mood}}
- director_notes: {{style (detailed performance direction), pacing (overall pace + max wps), accent (specific accent coaching)}}
- sample_context: a 1-2 sentence contextual setup so the TTS model enters the scene naturally
- transcript_with_tags: the FULL optimized script as ONE continuous block with inline [audio tags] choreographing every emotional shift, volume change, and delivery note (e.g. [whispers], [excitedly], [confident], [very slow], [sighs]). This is the exact text the TTS model will read.
- ambient_clip_volume: a float between 0.0 and 1.0 representing the volume of the original clip audio that should be kept as an ambient bed beneath the VO (e.g. 0.08 = 8%, 0.15 = 15%). Base this on the scene — outdoor/bustling environments warrant a higher ambient level (0.10–0.20), quiet/intimate/indoor scenes warrant a lower level (0.03–0.08). This is the raw multiplier that will be passed directly to FFmpeg.
- visual_breakdown: [{{index, start_sec, end_sec, emotion, volume_level, pacing_wps, action}}]
- word_count_math: [{{beat_index, available_sec, pacing_wps, max_words}}]
- slice_guide: [{{index, start_sec, end_sec, text (clean excerpt, no tags), aligns_with}}]"""


def _count_spoken_words(text: str) -> int:
    """Count spoken words, ignoring inline performance tags like [whispers]."""
    clean = re.sub(r"\[[^\]]+\]", " ", text or "")
    return len(re.findall(r"\b[\w']+\b", clean))


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class PersonaProfile(BaseModel):
    name: str
    age: str
    gender: str
    accent: str
    archetype: str
    rationale: str
    recommended_voice: str


class SceneBlock(BaseModel):
    location: str
    environment_details: str
    mood: str


class DirectorNotes(BaseModel):
    style: str
    pacing: str
    accent: str


class TranscriptBeat(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    emotion: str
    volume_level: str
    pacing_wps: float
    action: str


class SectionMath(BaseModel):
    beat_index: int
    available_sec: float
    pacing_wps: float
    max_words: int


class SliceGuide(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    text: str
    aligns_with: str


class VODirectorPlan(BaseModel):
    persona: PersonaProfile
    scene: SceneBlock
    director_notes: DirectorNotes
    sample_context: str
    transcript_with_tags: str
    ambient_clip_volume: float
    visual_breakdown: List[TranscriptBeat]
    word_count_math: List[SectionMath]
    slice_guide: List[SliceGuide]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VODirectorAgent:

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
        rough_cut_s3_key = state.get("rough_cut_s3_key")
        script_content = state.get("script_content", "")
        assembled_duration = state.get("assembled_duration", 0.0)
        edl = state.get("revised_edl") or state.get("edl", {})

        logger.info(
            f"Agent 5A VO Director starting: show_id={show_id}, ep={episode_number}, "
            f"assembled_duration={assembled_duration:.2f}s"
        )

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=5, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status: {e}")

        mongo_client = None
        s3_client = None
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

            if not rough_cut_s3_key:
                raise ValueError("rough_cut_s3_key is missing from state.")

            # 1. Download approved cut and upload to Gemini Files API
            signed_url = self._mint_presigned_url(s3_client, bucket_name, rough_cut_s3_key)
            local_path = os.path.join(self.tmp_dir, f"5a_cut_{os.path.basename(rough_cut_s3_key)}")
            logger.info(f"Downloading approved cut: {rough_cut_s3_key}")
            self._download(signed_url, local_path)

            logger.info("Uploading approved cut to Gemini Files API.")
            uploaded_file = genai_client.files.upload(file=local_path)
            uploaded_file = wait_for_gemini_file_active(genai_client, uploaded_file)

            # 2. Build clip-duration context from actual EDL trim points
            duration_lines = []
            for clip in edl.get("clips", []):
                trimmed = clip.get("trim_out_sec", 0.0) - clip.get("trim_in_sec", 0.0)
                duration_lines.append(
                    f"  Beat {clip['order']}: {clip.get('scene_action_name', 'clip')} "
                    f"— {trimmed:.2f}s "
                    f"(trim {clip.get('trim_in_sec', 0.0):.2f}s → {clip.get('trim_out_sec', 0.0):.2f}s)"
                )
            duration_context = (
                "FINAL EDIT CLIP DURATIONS "
                "(base ALL word-count math on these — ignore the raw script's timestamps):\n"
                + ("\n".join(duration_lines) if duration_lines else "  (no EDL data)")
                + f"\nTotal assembled duration: {assembled_duration:.2f}s"
            )
            max_total_words = max(8, int(assembled_duration * 1.45))
            target_total_words = max(6, int(assembled_duration * 1.25))
            duration_context += (
                "\n\nHARD VO LENGTH CONSTRAINT:"
                f"\n- The final video is {assembled_duration:.2f}s."
                f"\n- transcript_with_tags must contain at most {max_total_words} spoken words total "
                "(do not count bracketed audio tags as spoken words)."
                f"\n- Target {target_total_words} spoken words for natural TTS pacing."
                "\n- Prefer short sentence fragments over full-script coverage. Cut ideas ruthlessly."
                "\n- The VO must finish before the final frame; never write a transcript that requires more time."
            )

            # 3. Build contents
            contents = [
                AUDIO_DIRECTOR_SYSTEM_PROMPT + "\n\n" + DIRECTOR_JSON_RIDER,
                f"RAW SCRIPT (rough draft — all timestamps must be OVERRIDDEN by the video):\n{script_content}",
                duration_context,
                "APPROVED ROUGH CUT VIDEO:",
                uploaded_file,
            ]

            # 4. Director call
            logger.info("Calling gemini-3.1-pro-preview for VO direction.")
            response = self._call_with_retry(
                genai_client.models.generate_content,
                model="gemini-3.1-pro-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VODirectorPlan,
                ),
            )

            if not response or not response.parsed:
                raise ValueError("Director call returned an empty or unparseable response.")

            plan: VODirectorPlan = response.parsed
            plan_dict = plan.model_dump()
            spoken_words = _count_spoken_words(plan.transcript_with_tags)
            if spoken_words > max_total_words:
                logger.warning(
                    f"VO transcript too long ({spoken_words}>{max_total_words}); requesting shorter rewrite."
                )
                response = self._call_with_retry(
                    genai_client.models.generate_content,
                    model="gemini-3.1-pro-preview",
                    contents=contents + [
                        "\nREWRITE REQUIRED: Your previous transcript was too long. "
                        f"Return the same JSON schema again, but transcript_with_tags must be "
                        f"at most {max_total_words} spoken words and should target "
                        f"{target_total_words} spoken words. Preserve only the strongest ad beats."
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VODirectorPlan,
                    ),
                )
                if not response or not response.parsed:
                    raise ValueError("Director rewrite returned an empty or unparseable response.")
                plan = response.parsed
                plan_dict = plan.model_dump()
                spoken_words = _count_spoken_words(plan.transcript_with_tags)
                if spoken_words > max_total_words:
                    raise ValueError(
                        f"VO transcript is too long for {assembled_duration:.2f}s: "
                        f"{spoken_words} spoken words > max {max_total_words}."
                    )

            logger.info(
                f"5A Director complete. Persona: {plan.persona.name} ({plan.persona.archetype}), "
                f"recommended_voice={plan.persona.recommended_voice}, "
                f"transcript length={len(plan.transcript_with_tags)} chars, "
                f"spoken_words={spoken_words}/{max_total_words}."
            )

            # 5. Update state
            state["vo_director_plan"] = plan_dict
            state["current_agent"] = "agent5a"

            # 6. Persist to MongoDB
            if db is not None:
                try:
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=5,
                        status="completed",
                        output={"stage": "director", "vo_director_plan": plan_dict},
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 5A: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=5, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="agent_5b")
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after 5A: {e}")

            logger.info("Agent 5A VO Director completed successfully.")
            return state

        except Exception as e:
            logger.error(f"Agent 5A VO Director failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent5a", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=5, status="failed")
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


def agent_5a_director_node(state: Phase4State) -> Phase4State:
    agent = VODirectorAgent()
    return agent.process(state)
