import os
import uuid
import json
import logging
import subprocess
import shutil
import tempfile
from typing import Dict, Any, List
from datetime import datetime, timezone

import boto3
from pymongo import MongoClient

from app.services.final_assemblies_service import update_agent_output, append_versioned
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.workflow_state import Phase4State

logger = logging.getLogger(__name__)

__all__ = ["AssemblyAgent", "agent_2_assembly_node"]


class AssemblyAgent:
    def __init__(self):
        self.tmp_dir = os.environ.get("PHASE4_TMP_DIR", "/tmp/phase4")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.ffmpeg_path = os.environ.get("PHASE4_FFMPEG_PATH", "ffmpeg")
        self.ffprobe_path = os.environ.get("PHASE4_FFPROBE_PATH", "ffprobe")
        self.pipeline_service = PipelineService()

    def _mint_presigned_url(self, s3_client, bucket: str, s3_key: str) -> str:
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=86400 * 7,
        )

    def _probe_duration(self, filepath: str) -> float:
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def _assert_video_stream(self, filepath: str) -> None:
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,pix_fmt",
            "-of", "json",
            filepath,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout or "{}")
        streams = info.get("streams") or []
        if not streams:
            raise RuntimeError(f"Rough cut has no video stream: {filepath}")
        stream = streams[0]
        logger.info(
            "Rough cut video stream: "
            f"{stream.get('codec_name')} {stream.get('width')}x{stream.get('height')} "
            f"pix_fmt={stream.get('pix_fmt')}"
        )

    def _get_fit_filter(self, fit_mode: str, w: int, h: int) -> str:
        if fit_mode == "crop":
            return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        elif fit_mode == "pad":
            return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
        elif fit_mode == "blur_pad":
            return (f"split=2[original][copy];"
                    f"[copy]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:20[blurred];"
                    f"[original]scale={w}:{h}:force_original_aspect_ratio=decrease[scaled];"
                    f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2")
        return f"scale={w}:{h}"

    def process(self, state: Phase4State) -> Phase4State:
        show_id = state.get("show_id")
        episode_number = state.get("episode_number")
        episode_id = state.get("episode_id")
        job_id = state.get("job_id")
        title = state.get("title", "UNTITLED")
        title_safe = title.upper().replace(" ", "_").replace("/", "_")

        # Resolve EDL
        edl = state.get("revised_edl") or state.get("edl", {})
        clips = edl.get("clips", [])
        
        # Manifest
        clip_manifest = state.get("clip_manifest", [])
        by_key = {c["s3_key"]: c for g in clip_manifest for c in g.get("candidates", [])}

        target_res = os.environ.get("PHASE4_TARGET_RESOLUTION", "1080x1920")
        w, h = map(int, target_res.split("x"))
        target_fps = os.environ.get("PHASE4_TARGET_FPS", "30")
        fit_mode = os.environ.get("PHASE4_FIT_MODE", "crop")
        keep_clip_audio = True

        logger.info(f"Agent 2 Assembly starting for {show_id} Ep {episode_number}")

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=2, status="running")
            except:
                pass

        mongo_client = None
        s3_client = None

        workspace_dir = tempfile.mkdtemp(dir=self.tmp_dir, prefix="agent2_")
        
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

            # Sort EDL clips
            clips.sort(key=lambda x: x.get("order", 0))

            trimmed_info = []
            segment_paths = []
            any_dissolve = any(c.get("transition_to_next", "none").lower() != "none" for c in clips)

            # Download & Trim
            for i, clip in enumerate(clips):
                s3_key = clip.get("s3_key")
                if s3_key not in by_key:
                    err_msg = f"s3_key {s3_key} not in manifest"
                    state.setdefault("errors", []).append({"agent": "agent2", "error": err_msg})
                    raise ValueError(err_msg)

                manifest_clip = by_key[s3_key]
                trim_in = max(0.0, float(clip.get("trim_in_sec", 0.0)))
                trim_out = float(clip.get("trim_out_sec", 1.0))
                duration = max(0.4, trim_out - trim_in)

                url = self._mint_presigned_url(s3_client, bucket_name, s3_key)

                seg_filename = f"seg_{i:03d}.mp4"
                seg_path = os.path.join(workspace_dir, seg_filename)

                # Trim & Normalize
                has_audio = manifest_clip.get("has_audio", False)
                vf_expr = f"{self._get_fit_filter(fit_mode, w, h)},fps={target_fps},format=yuv420p,setsar=1"
                
                cmd = [
                    self.ffmpeg_path, "-y",
                    "-ss", str(trim_in),
                    "-t", str(duration),
                    "-i", url
                ]
                
                if not has_audio or not keep_clip_audio:
                    # Add silent track
                    cmd.extend(["-f", "lavfi", "-t", str(duration), "-i", "anullsrc=r=48000:cl=stereo"])
                    
                cmd.extend(["-filter_complex" if fit_mode == "blur_pad" else "-vf", vf_expr])

                cmd.extend(["-map", "0:v:0"])
                if not has_audio or not keep_clip_audio:
                    cmd.extend(["-map", "1:a"])
                else:
                    cmd.extend(["-map", "0:a?"])
                
                cmd.extend([
                    "-r", target_fps,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ar", "48000", "-ac", "2",
                    seg_path
                ])

                # Hack: for blur pad complex filter fixup
                if fit_mode == "blur_pad":
                    idx = cmd.index("-map") if "-map" in cmd else len(cmd)
                    # We rewrite the command to handle complex properly
                    cmd = [
                        self.ffmpeg_path, "-y",
                        "-ss", str(trim_in), "-t", str(duration),
                        "-i", url
                    ]
                    filter_str = f"[0:v]split=2[original][copy];[copy]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:20[blurred];[original]scale={w}:{h}:force_original_aspect_ratio=decrease[scaled];[blurred][scaled]overlay=(W-w)/2:(H-h)/2,fps={target_fps},format=yuv420p,setsar=1[vout]"
                    if not has_audio or not keep_clip_audio:
                        cmd.extend(["-f", "lavfi", "-t", str(duration), "-i", "anullsrc=r=48000:cl=stereo"])
                        cmd.extend(["-filter_complex", filter_str, "-map", "[vout]", "-map", "1:a"])
                    else:
                        cmd.extend(["-filter_complex", filter_str, "-map", "[vout]", "-map", "0:a"])

                    cmd.extend([
                        "-r", target_fps,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-ar", "48000", "-ac", "2",
                        seg_path
                    ])

                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    state.setdefault("errors", []).append({"agent": "agent2", "error": f"FFmpeg error: {e.stderr}"})
                    logger.error(f"FFmpeg trim failed: {e.stderr}")
                    raise

                segment_paths.append(seg_path)
                trimmed_info.append({
                    "order": i,
                    "s3_key": manifest_clip["s3_key"],
                    "version": manifest_clip["version"],
                    "trim_in_sec": trim_in,
                    "trim_out_sec": trim_out,
                    "duration": duration,
                    "transition_to_next": clip.get("transition_to_next", "none"),
                    "transition_duration_sec": clip.get("transition_duration_sec", 0.0)
                })

            final_rough_cut = os.path.join(workspace_dir, "rough_cut.mp4")

            if not any_dissolve:
                # Concat demuxer
                list_file = os.path.join(workspace_dir, "list.txt")
                with open(list_file, "w") as f:
                    for sp in segment_paths:
                        f.write(f"file '{sp}'\n")
                
                cmd = [
                    self.ffmpeg_path, "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", list_file,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-profile:v", "main",
                    "-level", "4.1",
                    "-c:a", "aac",
                    "-ar", "48000",
                    "-ac", "2",
                    "-b:a", "128k",
                    "-movflags", "+faststart",
                    final_rough_cut
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"FFmpeg concat failed: {e.stderr}")
            else:
                # Filter complex for dissolves
                inputs = []
                for sp in segment_paths:
                    inputs.extend(["-i", sp])
                
                filter_chains = []
                current_v = "[0:v]"
                current_a = "[0:a]"
                current_len = self._probe_duration(segment_paths[0])
                
                for i in range(len(segment_paths) - 1):
                    t_type = trimmed_info[i]["transition_to_next"]
                    t_dur = float(trimmed_info[i]["transition_duration_sec"])
                    next_v = f"[{i+1}:v]"
                    next_a = f"[{i+1}:a]"
                    out_v = f"[v{i+1}]"
                    out_a = f"[a{i+1}]"
                    next_len = self._probe_duration(segment_paths[i+1])
                    
                    if t_type.lower() != "none" and t_dur > 0:
                        offset = current_len - t_dur
                        filter_chains.append(f"{current_v}{next_v}xfade=transition=fade:duration={t_dur}:offset={offset}{out_v}")
                        filter_chains.append(f"{current_a}{next_a}acrossfade=d={t_dur}{out_a}")
                        current_len = current_len + next_len - t_dur
                    else:
                        # Hard cut via concat filter
                        filter_chains.append(f"{current_v}{current_a}{next_v}{next_a}concat=n=2:v=1:a=1{out_v}{out_a}")
                        current_len = current_len + next_len
                    
                    current_v = out_v
                    current_a = out_a
                
                filter_complex_str = ";".join(filter_chains)
                
                cmd = [self.ffmpeg_path, "-y"] + inputs + [
                    "-filter_complex", filter_complex_str,
                    "-map", current_v, "-map", current_a,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.1",
                    "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "128k",
                    "-movflags", "+faststart",
                    final_rough_cut
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"FFmpeg xfade concat failed: {e.stderr}")

            self._assert_video_stream(final_rough_cut)
            assembled_dur = self._probe_duration(final_rough_cut)

            # Versioning
            rough_cut_version = state.get("rough_cut_version", 0) + 1

            # episode_id can be None if the caller didn't set it — use a safe fallback
            ep_id = episode_id or f"ep{episode_number}"

            # Upload — use upload_file (multipart) to avoid loading the full video into memory
            s3_key = f"phase4/{show_id}/{ep_id}/rough_cut/v{rough_cut_version}.mp4"
            s3_client.upload_file(
                final_rough_cut,
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
                
            rough_cut_s3_url = self._mint_presigned_url(s3_client, bucket_name, s3_key)
            
            # State Update
            state["rough_cut_s3_key"] = s3_key
            state["rough_cut_s3_url"] = rough_cut_s3_url
            state["rough_cut_version"] = rough_cut_version
            state["assembled_duration"] = assembled_dur
            state["trimmed_clip_keys"] = trimmed_info
            state["current_agent"] = "agent2"

            if db is not None:
                try:
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=2,
                        status="completed",
                        output={
                            "rough_cut_s3_key": s3_key,
                            "rough_cut_version": rough_cut_version,
                            "duration": assembled_dur
                        }
                    )
                    append_versioned(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        array_field="rough_cuts",
                        entry={
                            "version": rough_cut_version,
                            "s3_key": s3_key,
                            "s3_url": rough_cut_s3_url,
                            "duration": assembled_dur,
                            "created_at": datetime.now(timezone.utc).isoformat()
                        }
                    )
                except Exception as e:
                    logger.warning(f"Agent 2 DB update error: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=2, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="agent_3")
                except:
                    pass

            return state

        except Exception as e:
            logger.error(f"Agent 2 Assembly failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent2", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=2, status="failed")
                except:
                    pass
            raise e

        finally:
            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)
            if mongo_client:
                mongo_client.close()


def agent_2_assembly_node(state: Phase4State) -> Phase4State:
    agent = AssemblyAgent()
    return agent.process(state)
