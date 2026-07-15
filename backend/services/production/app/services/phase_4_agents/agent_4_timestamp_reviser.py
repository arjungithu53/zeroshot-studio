import os
import time
import json
import logging
import requests
from datetime import datetime, timezone

import boto3
from pymongo import MongoClient
from google import genai
from google.genai import types

from app.services.final_assemblies_service import update_agent_output, append_versioned
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.gemini_files import wait_for_gemini_file_active
from app.services.phase_4_agents.workflow_state import Phase4State
from app.services.phase_4_agents.agent_1_edl_generator import (
    EditDecisionList,
    VersionChoice,
)

logger = logging.getLogger(__name__)

__all__ = ["TimestampReviserAgent", "agent_4_revise_node"]


TIMESTAMP_REVISER_SYSTEM_PROMPT = """You are an Expert DR (Direct Response) Video Editor performing a REVISION pass. A reviewer has assessed an assembled rough cut and returned a prioritized list of change requests. You will also receive the CURRENT Edit Decision List (the trims/order/transitions that produced the reviewed cut) and visual access to the raw clips.

YOUR JOB: produce an UPDATED Edit Decision List that resolves the reviewer's change requests while preserving everything that already works.

RULES:
1. Resolve the change requests in priority order. For each high/medium request, make the concrete edit (re-trim, drop a clip, reorder, change a transition, extend or tighten a beat). You may ignore a low-priority request only if honoring it would hurt pacing.
2. You may ONLY use the existing clips. You cannot request new footage. Reference clips by their exact file (clip_label + s3_key + version) exactly as in the current EDL/manifest. Multiple versions of the same shot may exist — never swap to a different version unless a change request explicitly calls for it.
3. Keep the same OUTPUT discipline as a first-pass EDL: every kept clip needs exact Trim IN/OUT in seconds, each justified by a specific visual cue, plus a transition (default Hard Cut) and a one-line reason. Re-verify the loop/ending: if the reviewer flagged a broken loop, either fix it with available frames or recommend a hard reset ending.
4. Be conservative: do not re-cut clips the reviewer praised. Change only what is needed.
5. State, in overall_strategy, exactly which change requests you addressed and how, and any you intentionally declined and why.

Return the full revised blueprint AND the JSON schema (same structure as a first-pass EDL), echoing exact clip_label/s3_key/version for every clip."""

REVISION_JSON_RIDER = """Return the full revised EDL as a single JSON object matching the provided schema. Echo exact clip_label/s3_key/version for every clip — never invent new keys or filenames. Timestamps in seconds as floats. transition_to_next is 'none' unless a dissolve is strictly justified."""


class TimestampReviserAgent:
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

    def process(self, state: Phase4State) -> Phase4State:
        show_id = state.get("show_id")
        episode_number = state.get("episode_number")
        job_id = state.get("job_id")
        clip_manifest = state.get("clip_manifest", [])
        current_edl = state.get("revised_edl") or state.get("edl", {})
        review_result = state.get("review_result", {})
        rough_cut_s3_key = state.get("rough_cut_s3_key")
        edit_loop_count = state.get("edit_loop_count", 0)

        logger.info(
            f"Agent 4 Timestamp Reviser starting: show_id={show_id}, ep={episode_number}, "
            f"edit_loop_count={edit_loop_count}"
        )

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=4, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status to running: {e}")

        mongo_client = None
        s3_client = None
        genai_client = None
        uploaded_files = []
        local_paths = []

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

            # Build by_key lookup and assign stable labels (same CLIP_A, CLIP_B, ...
            # ordering as Agent 1 so the model can cross-reference the current EDL)
            by_key: dict = {}
            labeled_candidates = []  # [{label, s3_key, version, shot_id, duration, filename, description}]

            idx = 0
            for group in clip_manifest:
                shot_id = group.get("shot_id")
                for candidate in group.get("candidates", []):
                    idx += 1
                    clip_label = f"CLIP_{chr(64 + idx)}" if idx <= 26 else f"CLIP_{idx}"
                    s3_key = candidate.get("s3_key")
                    by_key[s3_key] = {
                        "duration": candidate.get("duration", 0.0),
                        "shot_id": shot_id,
                        "label": clip_label,
                    }
                    labeled_candidates.append({
                        "label": clip_label,
                        "s3_key": s3_key,
                        "version": candidate.get("version"),
                        "shot_id": shot_id,
                        "duration": candidate.get("duration", 0.0),
                        "filename": candidate.get("filename", ""),
                        "description": candidate.get("description", ""),
                    })

            # 1. Download and upload all raw clips to Gemini Files API
            logger.info(f"Uploading {len(labeled_candidates)} raw clips to Gemini Files API.")
            clip_file_handles = []
            for cand in labeled_candidates:
                signed_url = self._mint_presigned_url(s3_client, bucket_name, cand["s3_key"])
                local_path = os.path.join(
                    self.tmp_dir,
                    f"rev_{cand['label']}_{os.path.basename(cand['s3_key'])}"
                )
                self._download(signed_url, local_path)
                local_paths.append(local_path)

                uploaded = genai_client.files.upload(file=local_path)
                uploaded = wait_for_gemini_file_active(genai_client, uploaded)
                uploaded_files.append(uploaded)
                clip_file_handles.append((cand, uploaded))

            # 2. Optionally upload the rough cut for visual reference
            rough_cut_handle = None
            if rough_cut_s3_key:
                try:
                    rc_url = self._mint_presigned_url(s3_client, bucket_name, rough_cut_s3_key)
                    rc_path = os.path.join(self.tmp_dir, f"rev_roughcut_{os.path.basename(rough_cut_s3_key)}")
                    self._download(rc_url, rc_path)
                    local_paths.append(rc_path)
                    rough_cut_handle = genai_client.files.upload(file=rc_path)
                    rough_cut_handle = wait_for_gemini_file_active(genai_client, rough_cut_handle)
                    uploaded_files.append(rough_cut_handle)
                    logger.info("Rough cut uploaded for visual reference.")
                except Exception as e:
                    logger.warning(f"Could not upload rough cut for reference: {e}")

            # 3. Build manifest text for model context
            manifest_lines = []
            for cand in labeled_candidates:
                manifest_lines.append(
                    f"  {cand['label']} | shot_id={cand['shot_id']} | "
                    f"s3_key={cand['s3_key']} | version={cand['version']} | "
                    f"duration={cand['duration']}s | filename={cand['filename']} | "
                    f"description={cand['description']}"
                )
            manifest_text = "AVAILABLE CLIPS (use ONLY these exact s3_keys):\n" + "\n".join(manifest_lines)

            change_requests = review_result.get("change_requests", [])
            change_requests_text = (
                "REVIEWER CHANGE REQUESTS (resolve in priority order):\n"
                + json.dumps(change_requests, indent=2)
            )

            current_edl_text = "CURRENT EDL (the trims/order/transitions that produced the reviewed cut):\n" + json.dumps(current_edl, indent=2)

            # 4. Build contents
            contents = [
                TIMESTAMP_REVISER_SYSTEM_PROMPT + "\n\n" + REVISION_JSON_RIDER,
                manifest_text,
                current_edl_text,
                change_requests_text,
            ]

            if rough_cut_handle:
                contents.append("CURRENT ROUGH CUT (for visual reference — do NOT copy from it, only reference timing):")
                contents.append(rough_cut_handle)

            contents.append("RAW CLIPS (re-cut from these):")
            for cand, file_handle in clip_file_handles:
                contents.append(
                    f"=== {cand['label']} (shot_id={cand['shot_id']}, "
                    f"s3_key={cand['s3_key']}, version={cand['version']}) ==="
                )
                contents.append(file_handle)

            # 5. Call model with retries
            backoff = [2, 4, 8]
            response = None
            for attempt in range(3):
                try:
                    response = genai_client.models.generate_content(
                        model="gemini-3.1-pro-preview",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=EditDecisionList,
                        ),
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"Gemini attempt {attempt + 1} failed: {e}. Retrying in {backoff[attempt]}s.")
                        time.sleep(backoff[attempt])
                    else:
                        raise

            if not response or not response.parsed:
                raise ValueError("Model returned an empty or unparseable response.")

            parsed: EditDecisionList = response.parsed

            # 6. Validate — drop any invented s3_keys, clamp trims
            valid_clips = []
            used_shot_ids: set = set()
            for clip in parsed.clips:
                if clip.s3_key not in by_key:
                    logger.warning(f"Invented s3_key '{clip.s3_key}' in revised EDL — dropping.")
                    state.setdefault("errors", []).append(
                        {"agent": "agent4", "error": f"Invalid s3_key {clip.s3_key} dropped from revision."}
                    )
                    continue

                clip_meta = by_key[clip.s3_key]
                shot_id = clip_meta["shot_id"]

                if shot_id in used_shot_ids:
                    logger.warning(f"Duplicate shot_id '{shot_id}' in revised EDL — dropping extra clip.")
                    state.setdefault("errors", []).append(
                        {"agent": "agent4", "error": f"Duplicate shot {shot_id} dropped from revision."}
                    )
                    continue

                used_shot_ids.add(shot_id)

                dur = clip_meta["duration"]
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
            parsed.clips = valid_clips

            # Backfill any missing version_choices for choose_one shots
            for group in clip_manifest:
                if group.get("selection_mode") == "choose_one":
                    shot_id = group.get("shot_id")
                    if not any(vc.shot_id == shot_id for vc in parsed.version_choices):
                        used = next(
                            (c for c in valid_clips if by_key.get(c.s3_key, {}).get("shot_id") == shot_id),
                            None,
                        )
                        if used:
                            parsed.version_choices.append(VersionChoice(
                                shot_id=shot_id,
                                chosen_s3_key=used.s3_key,
                                chosen_version=used.version,
                                considered=[c["s3_key"] for c in group.get("candidates", [])],
                                reason="Backfilled by system from used clip in revised EDL.",
                            ))

            revised_edl_dict = parsed.model_dump()

            # 7. Increment loop count and update state
            new_loop_count = edit_loop_count + 1
            state["revised_edl"] = revised_edl_dict
            state["edit_loop_count"] = new_loop_count
            state["current_agent"] = "agent4"

            logger.info(
                f"Agent 4 revision complete: {len(valid_clips)} clips, "
                f"edit_loop_count now={new_loop_count}"
            )

            # 8. Persist to MongoDB
            if db is not None:
                try:
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=4,
                        status="completed",
                        output={"revised_edl": revised_edl_dict, "loop": new_loop_count},
                    )
                    append_versioned(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        array_field="edl_versions",
                        entry={
                            "version": new_loop_count,
                            "edl": revised_edl_dict,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "kind": "revision",
                        },
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 4: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=4, status="completed")
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent="agent_2")
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after agent 4: {e}")

            return state

        except Exception as e:
            logger.error(f"Agent 4 Timestamp Reviser failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent4", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=4, status="failed")
                except Exception:
                    pass
            raise

        finally:
            if genai_client and uploaded_files:
                for f in uploaded_files:
                    try:
                        genai_client.files.delete(name=f.name)
                    except Exception as e:
                        logger.warning(f"Could not delete Gemini file {f.name}: {e}")
            for p in local_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {p}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_4_revise_node(state: Phase4State) -> Phase4State:
    agent = TimestampReviserAgent()
    return agent.process(state)
