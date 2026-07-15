import os
import re
import logging
import subprocess
import requests
from datetime import datetime, timezone
from typing import Dict

import boto3
from pymongo import MongoClient

from app.services.final_assemblies_service import update_agent_output
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.workflow_state import Phase4State

logger = logging.getLogger(__name__)

__all__ = ["DeliveryAgent", "agent_9_delivery_node"]


def _secs_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timecode: HH:MM:SS,mmm"""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(slice_guide: list, fallback_text: str, fallback_duration: float) -> str:
    """
    Build an SRT string from Agent 5A's slice_guide.
    Falls back to a single full-length cue if slice_guide is empty.
    """
    if not slice_guide:
        return (
            f"1\n"
            f"{_secs_to_srt_time(0.0)} --> {_secs_to_srt_time(fallback_duration)}\n"
            f"{fallback_text.strip()}\n"
        )

    cues = []
    for i, entry in enumerate(slice_guide, start=1):
        start = _secs_to_srt_time(float(entry.get("start_sec", 0.0)))
        end = _secs_to_srt_time(float(entry.get("end_sec", 0.0)))
        text = entry.get("text", "").strip()
        cues.append(f"{i}\n{start} --> {end}\n{text}")

    return "\n\n".join(cues) + "\n"


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


def _download(url: str, local_path: str) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


class DeliveryAgent:
    """
    Finalizes the ad for distribution.

    Generates a sidecar SRT file from Agent 5A's slice_guide (no burned-in captions),
    produces per-platform exports (stream copy of the final master), and writes
    final metadata to S3 + MongoDB.
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

        final_master_s3_key = state.get("final_master_s3_key")
        final_master_version = state.get("final_master_version", 1)
        assembled_duration = state.get("assembled_duration", 0.0)
        loudness_lufs = state.get("loudness_lufs", 0.0)
        clip_manifest = state.get("clip_manifest", [])
        target_platforms = state.get("target_platforms", ["reels", "tiktok", "shorts"])

        # Caption source: Agent 5A's slice_guide + transcript
        vo_director_plan = state.get("vo_director_plan", {})
        slice_guide = vo_director_plan.get("slice_guide", [])
        transcript = vo_director_plan.get("transcript_with_tags", state.get("script_content", ""))

        enable_exports = os.environ.get("PHASE4_ENABLE_EXPORTS", "true").lower() not in ("0", "false", "no")

        logger.info(
            f"Agent 9 Delivery starting: show_id={show_id}, ep={episode_number}, "
            f"platforms={target_platforms}, slice_guide_entries={len(slice_guide)}, "
            f"enable_exports={enable_exports}"
        )

        if not final_master_s3_key:
            raise ValueError("final_master_s3_key is missing from state — Agent 8 must run first.")

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=9, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status: {e}")

        mongo_client = None
        srt_local = None
        local_master = None
        local_exports = []

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

            title_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", title)[:50]
            ep_id = episode_id or f"ep{episode_number}"

            # 1. Build and upload SRT
            srt_content = _build_srt(slice_guide, transcript, assembled_duration)
            srt_filename = f"{title_safe}_v{final_master_version}.srt"
            srt_s3_key = f"phase4/{show_id}/{ep_id}/final/{srt_filename}"

            srt_local = os.path.join(self.tmp_dir, srt_filename)
            with open(srt_local, "w", encoding="utf-8") as f:
                f.write(srt_content)

            s3_client.upload_file(
                srt_local, bucket_name, srt_s3_key,
                ExtraArgs={"ContentType": "text/plain; charset=utf-8"},
            )
            captions_url = self._mint_presigned_url(s3_client, bucket_name, srt_s3_key)
            logger.info(f"SRT uploaded: {srt_s3_key} ({len(slice_guide)} cues)")

            # 2. Download final master (needed for platform exports)
            platform_exports: Dict[str, str] = {}
            if enable_exports and target_platforms:
                local_master = os.path.join(
                    self.tmp_dir,
                    f"9_master_{os.path.basename(final_master_s3_key)}",
                )
                logger.info(f"Downloading final master for exports: {final_master_s3_key}")
                signed = self._mint_presigned_url(s3_client, bucket_name, final_master_s3_key)
                _download(signed, local_master)

                # 3. Per-platform exports — stream copy (all are 9:16, same spec)
                for platform in target_platforms:
                    export_filename = f"{title_safe}_FINAL_{platform}_v{final_master_version}.mp4"
                    export_s3_key = f"phase4/{show_id}/{ep_id}/final/{export_filename}"
                    local_export = os.path.join(self.tmp_dir, export_filename)
                    local_exports.append(local_export)

                    _run_ffmpeg([
                        "ffmpeg", "-y",
                        "-i", local_master,
                        "-c", "copy",
                        "-movflags", "+faststart",
                        local_export,
                    ])

                    s3_client.upload_file(
                        local_export, bucket_name, export_s3_key,
                        ExtraArgs={"ContentType": "video/mp4"},
                    )
                    platform_exports[platform] = self._mint_presigned_url(
                        s3_client, bucket_name, export_s3_key
                    )
                    logger.info(f"Export uploaded [{platform}]: {export_s3_key}")

            # 4. Assemble final metadata
            final_metadata = {
                "title": title,
                "episode_id": ep_id,
                "show_id": show_id,
                "assembled_duration_sec": assembled_duration,
                "loudness_lufs": loudness_lufs,
                "clip_count": sum(len(g.get("candidates", [])) for g in clip_manifest),
                "final_master_s3_key": final_master_s3_key,
                "captions_s3_key": srt_s3_key,
                "captions_cue_count": len(slice_guide) or 1,
                "platform_exports": platform_exports,
                "target_platforms": target_platforms,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # 5. Update state
            state["captions_s3_key"] = srt_s3_key
            state["platform_exports"] = platform_exports
            state["final_metadata"] = final_metadata
            state["current_agent"] = "agent9"
            state["pipeline_status"] = "completed"

            # 6. Persist to MongoDB
            if db is not None:
                try:
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=9,
                        status="completed",
                        output=final_metadata,
                    )
                    db["final_assemblies"].update_one(
                        {"show_id": show_id, "episode_number": episode_number},
                        {
                            "$set": {
                                "deliverables": {
                                    "captions_srt": srt_s3_key,
                                    "exports": platform_exports,
                                },
                                "final_metadata": final_metadata,
                                "pipeline_status": "completed",
                            }
                        },
                        upsert=True,
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 9: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=9, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="completed")
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=9, status="pipeline_complete")
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after agent 9: {e}")

            logger.info("Agent 9 Delivery completed successfully. Pipeline complete.")
            return state

        except Exception as e:
            logger.error(f"Agent 9 Delivery failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent9", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=9, status="failed")
                except Exception:
                    pass
            raise

        finally:
            for path in ([srt_local] if srt_local else []) + \
                        ([local_master] if local_master else []) + \
                        local_exports:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {path}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_9_delivery_node(state: Phase4State) -> Phase4State:
    agent = DeliveryAgent()
    return agent.process(state)
