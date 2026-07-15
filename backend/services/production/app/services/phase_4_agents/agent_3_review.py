import os
import json
import time
import logging
import tempfile
import requests
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel

import boto3
from pymongo import MongoClient
from google import genai
from google.genai import types

from app.services.final_assemblies_service import update_agent_output, append_versioned
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.gemini_files import wait_for_gemini_file_active
from app.services.phase_4_agents.workflow_state import Phase4State

logger = logging.getLogger(__name__)

__all__ = ["FinalCutReviewAgent", "agent_3_review_node", "route_after_review"]


FINAL_CUT_REVIEW_SYSTEM_PROMPT = """You are a Senior Direct-Response (DR) Creative Director and Final-Cut QC reviewer for short-form vertical ads (TikTok, Instagram Reels, YouTube Shorts). You are reviewing an ASSEMBLED ROUGH CUT (no final voiceover or music yet). Your job is to decide whether the cut is ready to advance to voiceover, or whether it needs another editing pass.

GUIDING PRINCIPLES:
1. The script and shot list are GUIDELINES, not rules. Judge the literal video in front of you. Reward choices that improve hook rate and retention even when they diverge from the script; never demand fidelity to the script for its own sake.
2. Retention is the metric. Evaluate: Is the first 1–2 seconds a strong hook? Does momentum hold with no dead air? Are any two clips redundant (same action twice)? Are cuts snappy but not glitchy (clips generally 2–5s)? Does the ending land, and if a loop was intended, does the last frame actually connect to the first (angle, posture, lighting, framing)?
3. Be specific and actionable. If you request an edit, tie each requested change to a concrete operation the editor can perform: trim tighter, extend a beat, drop a clip, reorder, change a transition, or fix a broken loop. Reference clips by their on-screen action and, where possible, approximate timecodes in the assembled cut.
4. Do not invent footage. Only request changes achievable by re-trimming, reordering, dropping, or re-transitioning the EXISTING clips. You cannot ask for new shots.
5. Calibrate strictly. Approve only if the cut would credibly perform as a DR ad. Otherwise request an edit. A cut that is "fine" but has obvious dead air, a weak hook, or a redundant beat should be sent back.

DECISION: output exactly one of "approved" or "edit".
- "approved": the cut is ready for voiceover.
- "edit": provide a prioritized, concrete change list.

Always return your assessment in the required JSON structure: a decision, an overall score (0–100), strengths, issues, and—if editing—an ordered list of change requests each with {target (which clip/beat), problem, fix (the concrete edit), and priority}."""

REVIEW_JSON_RIDER = """Return JSON matching the provided schema. decision must be exactly 'approved' or 'edit'. change_requests may be an empty list if decision is 'approved'."""


class ChangeRequest(BaseModel):
    target: str
    problem: str
    fix: str
    priority: str  # "high" | "medium" | "low"


class FinalCutReview(BaseModel):
    decision: str
    overall_score: int
    hook_assessment: str
    pacing_assessment: str
    strengths: List[str]
    issues: List[str]
    loop_status: str  # "ok" | "broken" | "n/a"
    change_requests: List[ChangeRequest]


class FinalCutReviewAgent:
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
        rough_cut_s3_key = state.get("rough_cut_s3_key")
        rough_cut_version = state.get("rough_cut_version", 0)
        edit_loop_count = state.get("edit_loop_count", 0)

        logger.info(
            f"Agent 3 Final-Cut Review starting: show_id={show_id}, ep={episode_number}, "
            f"rough_cut_version={rough_cut_version}, edit_loop_count={edit_loop_count}"
        )

        if job_id:
            try:
                self.pipeline_service.update_job_status(job_id=job_id, agent_number=3, status="running")
            except Exception as e:
                logger.warning(f"Could not update pipeline status to running: {e}")

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

            # 1. Download the rough cut via a fresh presigned URL
            if not rough_cut_s3_key:
                raise ValueError("rough_cut_s3_key is missing from state.")

            signed_url = self._mint_presigned_url(s3_client, bucket_name, rough_cut_s3_key)
            filename = os.path.basename(rough_cut_s3_key)
            local_path = os.path.join(self.tmp_dir, f"review_{filename}")

            logger.info(f"Downloading rough cut from s3_key={rough_cut_s3_key}")
            with requests.get(signed_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # 2. Upload to Gemini Files API
            logger.info("Uploading rough cut to Gemini Files API.")
            uploaded_file = genai_client.files.upload(file=local_path)
            uploaded_file = wait_for_gemini_file_active(genai_client, uploaded_file)

            # 3. Build a compact shot list summary for context
            shot_list = state.get("shot_list", {})
            edl = state.get("revised_edl") or state.get("edl", {})
            script_content = state.get("script_content", "")

            edl_summary = ""
            if edl.get("clips"):
                lines = [f"  Clip {c['order']}: {c.get('scene_action_name', '')} "
                         f"[{c.get('trim_in_sec', 0):.2f}s → {c.get('trim_out_sec', 0):.2f}s]"
                         for c in edl["clips"]]
                edl_summary = "EDL CONTEXT (clips used in this cut):\n" + "\n".join(lines)

            contents = [
                FINAL_CUT_REVIEW_SYSTEM_PROMPT + "\n\n" + REVIEW_JSON_RIDER,
                f"SCRIPT (GUIDELINE ONLY — do not demand fidelity):\n{script_content}",
                f"SHOT LIST SUMMARY (GUIDELINE ONLY):\n{json.dumps(shot_list, indent=2)}",
                edl_summary,
                "ASSEMBLED ROUGH CUT VIDEO:",
                uploaded_file,
            ]

            # 4. Call model with retries
            backoff = [2, 4, 8]
            response = None
            for attempt in range(3):
                try:
                    response = genai_client.models.generate_content(
                        model="gemini-3.1-pro-preview",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=FinalCutReview,
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

            parsed: FinalCutReview = response.parsed

            # Normalise decision to lowercase and guard against unexpected values
            decision = parsed.decision.strip().lower()
            if decision not in ("approved", "edit"):
                logger.warning(f"Unexpected decision value '{decision}'. Defaulting to 'edit'.")
                decision = "edit"
            parsed.decision = decision

            review_dict = parsed.model_dump()

            # 5. Update state
            state["review_result"] = review_dict
            state["review_decision"] = decision
            state["current_agent"] = "agent3"

            logger.info(
                f"Agent 3 review complete: decision={decision}, score={parsed.overall_score}, "
                f"change_requests={len(parsed.change_requests)}"
            )

            # 6. Persist to MongoDB
            if db is not None:
                try:
                    update_agent_output(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        agent_number=3,
                        status="completed",
                        output=review_dict,
                    )
                    append_versioned(
                        db=db,
                        show_id=show_id,
                        episode_number=episode_number,
                        array_field="reviews",
                        entry={
                            "for_version": rough_cut_version,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            **review_dict,
                        },
                    )
                except Exception as e:
                    logger.warning(f"DB update failed in agent 3: {e}")

            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=3, status="completed")
                    next_agent = "agent_4" if decision == "edit" and edit_loop_count < 2 else "agent_5"
                    self.pipeline_service.update_job_current_agent(job_id=job_id, current_agent=next_agent)
                except Exception as e:
                    logger.warning(f"Could not update pipeline status after agent 3: {e}")

            return state

        except Exception as e:
            logger.error(f"Agent 3 Final-Cut Review failed: {e}", exc_info=True)
            state.setdefault("errors", []).append({"agent": "agent3", "error": str(e)})
            state["pipeline_status"] = "failed"
            if job_id:
                try:
                    self.pipeline_service.update_job_status(job_id=job_id, agent_number=3, status="failed")
                except Exception:
                    pass
            raise

        finally:
            if genai_client and uploaded_file:
                try:
                    genai_client.files.delete(name=uploaded_file.name)
                except Exception as e:
                    logger.warning(f"Could not delete uploaded file from Gemini Files API: {e}")
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {local_path}: {e}")
            if mongo_client:
                mongo_client.close()


def agent_3_review_node(state: Phase4State) -> Phase4State:
    agent = FinalCutReviewAgent()
    return agent.process(state)


def route_after_review(state: Phase4State) -> str:
    """
    Conditional edge function for the LangGraph workflow.

    Returns "revise" if the review decision is "edit" and the loop cap has not
    been reached; returns "vo" in all other cases (approved, or force-pass when
    edit_loop_count >= 2).
    """
    decision = state.get("review_decision", "approved")
    edit_loop_count = state.get("edit_loop_count", 0)

    if decision == "edit" and edit_loop_count < 2:
        logger.info(f"route_after_review → revise (loop {edit_loop_count + 1}/2)")
        return "revise"

    if decision == "edit" and edit_loop_count >= 2:
        logger.info("route_after_review → vo (force-pass: max loops reached)")
    else:
        logger.info("route_after_review → vo (approved)")

    return "vo"
