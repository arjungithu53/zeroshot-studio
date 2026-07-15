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

__all__ = ["AVMergeAgent", "agent_6_av_merge_node"]


def _run_ffmpeg(cmd: list) -> None:
    """Run an ffmpeg command, streaming stderr to logs, raising on non-zero exit."""
    logger.info(f"FFmpeg: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    for line in proc.stderr:
        line = line.rstrip()
        if line:
            logger.debug(f"ffmpeg | {line}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")


def _download(url: str, local_path: str) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


class AVMergeAgent:
    """Lays the single-take VO over the approved cut, producing a VO preview MP4."""

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
        rough_cut_s3_key = state.get("rough_cut_s3_key")
        vo_s3_key = state.get("vo_s3_key")
        vo_director_plan = state.get("vo_director_plan", {})

        # Default True — original clip audio is kept as an ambient bed
        env_override = os.environ.get("PHASE4_KEEP_CLIP_AUDIO", "").lower()
        keep_clip_audio = env_override not in ("0", "false", "no") if env_override else True

        # Volume decided by Agent 5A; clamp to a safe range [0.01, 1.0]
        ambient_volume = float(vo_director_plan.get("ambient_clip_volume", 0.08))
        ambient_volume = max(0.01, min(ambient_volume, 1.0))

        logger.info(
            f"Agent 6 A/V Merge starting: show_id={show_id}, ep={episode_number}, "
            f"keep_clip_audio={keep_clip_audio}, ambient_volume={ambient_volume}"
        )

        if not rough_cut_s3_key:
            raise ValueError("rough_cut_s3_key is missing from state.")
        if not vo_s3_key:
            raise ValueError("vo_s3_key is missing from state.")

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=6, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status: {e}")

        mongo_client = None
        local_cut = None
        local_vo = None
        local_preview = None

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

            # 1. Download both source files via fresh presigned URLs
            cut_url = self._mint_presigned_url(s3_client, bucket_name, rough_cut_s3_key)
            vo_url = self._mint_presigned_url(s3_client, bucket_name, vo_s3_key)

            local_cut = os.path.join(self.tmp_dir, f"6_cut_{os.path.basename(rough_cut_s3_key)}")
            local_vo = os.path.join(self.tmp_dir, f"6_vo_{os.path.basename(vo_s3_key)}")

            logger.info(f"Downloading approved cut: {rough_cut_s3_key}")
            _download(cut_url, local_cut)
            logger.info(f"Downloading VO: {vo_s3_key}")
            _download(vo_url, local_vo)

            # 2. Determine version and output path
            vo_preview_version = state.get("vo_preview_version", 0) + 1
            ep_id = episode_id or f"ep{episode_number}"
            preview_filename = f"v{vo_preview_version}.mp4"
            local_preview = os.path.join(self.tmp_dir, f"6_preview_{preview_filename}")

            # 3. FFmpeg merge
            if not keep_clip_audio:
                # Mute clip audio entirely; VO is the sole program track
                cmd = [
                    "ffmpeg", "-y",
                    "-i", local_cut,
                    "-i", local_vo,
                    "-map", "0:v",
                    "-map", "1:a",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest",
                    "-movflags", "+faststart",
                    local_preview,
                ]
            else:
                # Mix original clip audio as ambient bed (volume set by Agent 5A) + VO
                cmd = [
                    "ffmpeg", "-y",
                    "-i", local_cut,
                    "-i", local_vo,
                    "-filter_complex",
                    f"[0:a]volume={ambient_volume}[amb];[amb][1:a]amix=inputs=2:duration=first:weights=1 1:normalize=0[a]",
                    "-map", "0:v",
                    "-map", "[a]",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    local_preview,
                ]

            _run_ffmpeg(cmd)
            logger.info(f"VO preview rendered: {local_preview}")

            # 4. Upload to S3
            s3_key = f"phase4/{show_id}/{ep_id}/with_vo/{preview_filename}"
            s3_client.upload_file(
                local_preview,
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            signed_url = self._mint_presigned_url(s3_client, bucket_name, s3_key)
            logger.info(f"VO preview uploaded to S3: {s3_key}")

            # 5. Update state
            state["vo_preview_s3_key"] = s3_key
            state["vo_preview_s3_url"] = signed_url
            state["vo_preview_version"] = vo_preview_version
            state["current_agent"] = "agent6"

            # 6. Persist to MongoDB
            if db is not None:
                try:
                    preview_entry = {
                        "version": vo_preview_version,
                        "s3_key": s3_key,
                        "s3_url": signed_url,
                        "keep_clip_audio": keep_clip_audio,
                        "ambient_clip_volume": ambient_volume,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=6,
                        status="completed",
                        output=preview_entry,
                    )
                    append_versioned(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        array_field="vo_preview",
                        entry=preview_entry,
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 6: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=6, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="agent_7")
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after agent 6: {e}")

            logger.info("Agent 6 A/V Merge completed successfully.")
            return state

        except Exception as e:
            logger.error(f"Agent 6 A/V Merge failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent6", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=6, status="failed")
                except Exception:
                    pass
            raise

        finally:
            for path in (local_cut, local_vo, local_preview):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {path}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_6_av_merge_node(state: Phase4State) -> Phase4State:
    agent = AVMergeAgent()
    return agent.process(state)
