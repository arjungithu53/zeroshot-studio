import os
import re
import time
import logging
import subprocess
import requests
from datetime import datetime, timezone

import boto3
from pymongo import MongoClient

from app.services.final_assemblies_service import update_agent_output, append_versioned
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.workflow_state import Phase4State

logger = logging.getLogger(__name__)

__all__ = ["FinalMixAgent", "agent_8_final_mix_node", "route_after_mix"]

# Tunable sidechain / mix constants
_SC_THRESHOLD = 0.05
_SC_RATIO = 8
_SC_ATTACK = 15
_SC_RELEASE = 300
_SC_MAKEUP = 2


def _build_volume_expr(envelope: list, fallback: float = 0.15) -> str:
    """
    Convert Agent 7A's volume_envelope into a nested FFmpeg if(between(t,...)) expression.

    Example output for two sections:
        if(between(t,0.0,8.0),0.15,if(between(t,8.0,20.0),0.45,0.2))
    """
    if not envelope:
        return str(fallback)

    sorted_env = sorted(envelope, key=lambda s: s["start_sec"])
    # Base case: last section's volume (catch-all for any gap past the last entry)
    expr = str(sorted_env[-1]["volume"])
    # Build outward from second-to-last to first
    for section in reversed(sorted_env[:-1]):
        expr = (
            f"if(between(t,{section['start_sec']},{section['end_sec']}),"
            f"{section['volume']},{expr})"
        )
    return expr


def _run_ffmpeg(cmd: list) -> None:
    logger.info(f"FFmpeg: {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    for line in proc.stderr:
        line = line.rstrip()
        if line:
            logger.debug(f"ffmpeg | {line}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")


def _measure_loudness(path: str) -> float:
    """Run an ebur128 pass and return integrated loudness in LUFS."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
        # Parse "I: -14.3 LUFS" from stderr
        match = re.search(r"I:\s+([-\d.]+)\s+LUFS", result.stderr)
        if match:
            return float(match.group(1))
    except Exception as e:
        logger.warning(f"Loudness measurement failed: {e}")
    return float(os.environ.get("PHASE4_TARGET_LUFS", "-14"))


def _download(url: str, local_path: str) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


class FinalMixAgent:
    """
    Combines visuals + VO + music into the finished ad master.

    Filter chain:
      1. Apply Agent 7A's section-level volume envelope to the music stem.
      2. Sidechain-compress the music keyed by the VO (ducks under narration).
      3. Mix VO + ducked music via amix.
      4. Loudness-normalize to target LUFS for social platforms.
      5. Mux with the video stream from the VO-preview and upload to S3.
    """

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
        episode_id = state.get("episode_id")
        episode_number = state.get("episode_number")
        job_id = state.get("job_id")
        title = state.get("title", "untitled")

        vo_preview_s3_key = state.get("vo_preview_s3_key")
        vo_s3_key = state.get("vo_s3_key")
        music_s3_key = state.get("music_s3_key")
        rough_cut_s3_key = state.get("rough_cut_s3_key")

        # Volume envelope from Agent 7A (section-level mix weights)
        music_director_plan = state.get("music_director_plan", {})
        volume_envelope = music_director_plan.get("volume_envelope", [])

        # Ambient clip audio volume from Agent 5A (raw multiplier, e.g. 0.08)
        vo_director_plan = state.get("vo_director_plan", {})
        ambient_clip_volume = float(vo_director_plan.get("ambient_clip_volume", 0.08))
        ambient_clip_volume = max(0.01, min(ambient_clip_volume, 1.0))

        lufs = os.environ.get("PHASE4_TARGET_LUFS", "-14")
        tp = os.environ.get("PHASE4_TARGET_TP", "-1.5")

        logger.info(
            f"Agent 8 Final Mix starting: show_id={show_id}, ep={episode_number}, "
            f"target={lufs} LUFS / {tp} dBTP, "
            f"volume_envelope sections={len(volume_envelope)}, "
            f"ambient_clip_volume={ambient_clip_volume}"
        )

        if not vo_preview_s3_key:
            raise ValueError("vo_preview_s3_key is missing from state.")
        if not vo_s3_key:
            raise ValueError("vo_s3_key is missing from state.")
        if not music_s3_key:
            raise ValueError("music_s3_key is missing from state.")
        if not rough_cut_s3_key:
            raise ValueError("rough_cut_s3_key is missing from state.")

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=8, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status: {e}")

        mongo_client = None
        local_preview = local_vo = local_music = local_cut = local_final = None

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

            # 1. Download all three stems
            final_version = state.get("final_master_version", 0) + 1
            title_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", title)[:50]
            ep_id = episode_id or f"ep{episode_number}"

            local_preview = os.path.join(self.tmp_dir, f"8_preview_{os.path.basename(vo_preview_s3_key)}")
            local_vo = os.path.join(self.tmp_dir, f"8_vo_{os.path.basename(vo_s3_key)}")
            local_music = os.path.join(self.tmp_dir, f"8_music_{os.path.basename(music_s3_key)}")
            local_cut = os.path.join(self.tmp_dir, f"8_cut_{os.path.basename(rough_cut_s3_key)}")
            local_final = os.path.join(self.tmp_dir, f"8_final_{title_safe}_v{final_version}.mp4")

            for s3_key, local_path, label in (
                (vo_preview_s3_key, local_preview, "VO preview"),
                (vo_s3_key, local_vo, "VO stem"),
                (music_s3_key, local_music, "music stem"),
                (rough_cut_s3_key, local_cut, "rough cut (ambient)"),
            ):
                logger.info(f"Downloading {label}: {s3_key}")
                signed = self._mint_presigned_url(s3_client, bucket_name, s3_key)
                _download(signed, local_path)

            # 2. Build dynamic volume expression from Agent 7A's envelope
            vol_expr = _build_volume_expr(volume_envelope)
            logger.info(f"Music volume expression: {vol_expr[:120]}{'...' if len(vol_expr) > 120 else ''}")

            # 3. Build filter_complex
            #
            # Inputs:
            #   0 = vo_preview.mp4  (video only — 0:v mapped)
            #   1 = vo.wav          (VO stem — used as sidechain key AND program audio)
            #   2 = music.wav       (music stem)
            #   3 = rough_cut.mp4   (ambient clip audio at Agent 5A's volume)
            #
            # Chain:
            #   amb_fmt   → normalize format/rate, apply ambient volume
            #   mus_fmt   → normalize format/rate
            #   vo_fmt    → normalize format/rate → split into vo_sc (sidechain) + vo_mix (program)
            #   mus_vol   → apply Agent 7A section-level volume envelope
            #   mus_ducked→ sidechain compress (keyed by vo_sc)
            #   [vo_mix + mus_ducked + amb_fmt] → amix (normalize=0) → loudnorm → aout
            #
            # normalize=0 on amix: prevents FFmpeg from halving levels when mixing 3 streams.
            # loudnorm handles the final integrated loudness target.
            filter_complex = (
                f"[3:a]aformat=channel_layouts=stereo,aresample=48000,"
                f"volume={ambient_clip_volume}[amb_fmt];"
                f"[2:a]aformat=channel_layouts=stereo,aresample=48000[mus_fmt];"
                f"[1:a]aformat=channel_layouts=stereo,aresample=48000[vo_fmt];"
                f"[vo_fmt]asplit=2[vo_sc][vo_mix];"
                f"[mus_fmt]volume=eval=frame:volume='{vol_expr}'[mus_vol];"
                f"[mus_vol][vo_sc]sidechaincompress="
                f"threshold={_SC_THRESHOLD}:ratio={_SC_RATIO}:"
                f"attack={_SC_ATTACK}:release={_SC_RELEASE}:makeup={_SC_MAKEUP}[mus_ducked];"
                f"[vo_mix][mus_ducked][amb_fmt]amix=inputs=3:duration=first:"
                f"weights=1 1 1:normalize=0:dropout_transition=0[mix];"
                f"[mix]loudnorm=I={lufs}:TP={tp}:LRA=11[aout]"
            )

            cmd = [
                "ffmpeg", "-y",
                "-i", local_preview,
                "-i", local_vo,
                "-i", local_music,
                "-i", local_cut,
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart",
                local_final,
            ]
            _run_ffmpeg(cmd)
            logger.info(f"Final master rendered: {local_final}")

            # 4. Measure actual loudness
            loudness_lufs = _measure_loudness(local_final)
            logger.info(f"Measured integrated loudness: {loudness_lufs:.1f} LUFS")

            # 5. Upload to S3
            final_filename = f"{title_safe}_FINAL_v{final_version}.mp4"
            s3_key = f"phase4/{show_id}/{ep_id}/final/{final_filename}"
            s3_client.upload_file(
                local_final,
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            signed_url = self._mint_presigned_url(s3_client, bucket_name, s3_key)
            logger.info(f"Final master uploaded to S3: {s3_key}")

            # 6. Update state
            state["final_master_s3_key"] = s3_key
            state["final_master_s3_url"] = signed_url
            state["final_master_version"] = final_version
            state["loudness_lufs"] = loudness_lufs
            state["current_agent"] = "agent8"

            # 7. Persist to MongoDB
            enable_delivery = os.environ.get("PHASE4_ENABLE_DELIVERY", "true").lower() not in ("0", "false", "no")

            if db is not None:
                try:
                    master_entry = {
                        "version": final_version,
                        "s3_key": s3_key,
                        "s3_url": signed_url,
                        "loudness_lufs": loudness_lufs,
                        "target_lufs": float(lufs),
                        "target_tp": float(tp),
                        "volume_envelope_sections": len(volume_envelope),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=8,
                        status="completed",
                        output=master_entry,
                    )
                    append_versioned(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        array_field="final_masters",
                        entry=master_entry,
                    )
                    if not enable_delivery:
                        # No Agent 9 — mark the whole pipeline complete here
                        db["final_assemblies"].update_one(
                            {"show_id": show_id, "episode_number": episode_number},
                            {"$set": {"pipeline_status": "completed"}},
                        )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 8: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=8, status="completed")
                    next_agent = "agent_9" if enable_delivery else "completed"
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent=next_agent)
                    if not enable_delivery:
                        self.pipeline_service.update_job_status(job_id=job_id, agent_number=8, status="pipeline_complete")
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after agent 8: {e}")

            logger.info("Agent 8 Final Mix completed successfully.")
            return state

        except Exception as e:
            logger.error(f"Agent 8 Final Mix failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent8", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=8, status="failed")
                except Exception:
                    pass
            raise

        finally:
            for path in (local_preview, local_vo, local_music, local_cut, local_final):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {path}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_8_final_mix_node(state: Phase4State) -> Phase4State:
    agent = FinalMixAgent()
    return agent.process(state)


def route_after_mix(state: Phase4State) -> str:
    """Conditional edge: → 'deliver' if Agent 9 is enabled, else → 'end'."""
    enable_delivery = os.environ.get("PHASE4_ENABLE_DELIVERY", "true").lower() not in ("0", "false", "no")
    return "deliver" if enable_delivery else "end"
