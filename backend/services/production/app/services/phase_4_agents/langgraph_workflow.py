"""
Phase 4 LangGraph workflow — initialize + full graph wiring.

Entry:   initialize_node  (builds clip_manifest from Agent 0 human selections)
Exit:    agent_9_delivery_node → END  (or agent_8 → END if delivery disabled)

Agent splits vs the original spec:
  agent_5_vo_node   →  agent_5a_director_node → agent_5b_tts_node
  agent_7_music_node → agent_7a_music_director_node → agent_7b_lyria_node
"""

import json
import os
import re
import subprocess
import logging
import requests
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
from bson import ObjectId
from pymongo import MongoClient
from langgraph.graph import StateGraph, END

from app.services.final_assemblies_service import upsert_assembly, update_agent_output
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.workflow_state import Phase4State, ClipRef, ShotCandidates

# ── node imports ─────────────────────────────────────────────────────────────
from app.services.phase_4_agents.agent_1_edl_generator import agent_1_edl_node
from app.services.phase_4_agents.agent_2_assembly import agent_2_assembly_node
from app.services.phase_4_agents.agent_3_review import agent_3_review_node, route_after_review
from app.services.phase_4_agents.agent_4_timestamp_reviser import agent_4_revise_node
from app.services.phase_4_agents.agent_5a_director import agent_5a_director_node
from app.services.phase_4_agents.agent_5b_tts import agent_5b_tts_node
from app.services.phase_4_agents.agent_6_av_merge import agent_6_av_merge_node
from app.services.phase_4_agents.agent_7a_music_director import agent_7a_music_director_node
from app.services.phase_4_agents.agent_7b_lyria import agent_7b_lyria_node
from app.services.phase_4_agents.agent_8_final_mix import agent_8_final_mix_node, route_after_mix
from app.services.phase_4_agents.agent_9_delivery import agent_9_delivery_node

logger = logging.getLogger(__name__)

__all__ = ["initialize_node", "build_phase4_graph", "run_phase4_pipeline"]

_TMP_DIR = os.environ.get("PHASE4_TMP_DIR", "/tmp/phase4")
_SHOTS_COLLECTION = os.environ.get("PHASE4_SHOTS_COLLECTION", "shots")
_MOVIES_COLLECTION = os.environ.get("PHASE4_MOVIES_COLLECTION", "movies")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_s3_key(presigned_url: str, bucket_name: str) -> str:
    """Strip host prefix and query string from a presigned URL → durable s3_key."""
    parsed = urlparse(presigned_url)
    path = parsed.path.lstrip("/")
    prefix = bucket_name.rstrip("/") + "/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def _mint_presigned_url(s3_client: Any, bucket: str, s3_key: str) -> str:
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=86400 * 7,
    )


def _ffprobe_clip(local_path: str) -> tuple[float, bool]:
    """Return (duration_seconds, has_audio) via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,duration",
                "-of", "json",
                local_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        duration = 0.0
        has_audio = False
        for s in streams:
            if s.get("codec_type") == "video" and not duration:
                duration = float(s.get("duration", 0.0) or 0.0)
            if s.get("codec_type") == "audio":
                has_audio = True
        # Fallback: probe format duration
        if not duration:
            fmt_result = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1",
                 local_path],
                capture_output=True, text=True, timeout=30,
            )
            duration = float(fmt_result.stdout.strip() or 0.0)
        return duration, has_audio
    except Exception as e:
        logger.warning(f"ffprobe failed for {local_path}: {e}")
        return 0.0, False


def _download_clip(url: str, local_path: str) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def _parse_scene_shot_version(s3_key: str) -> tuple[int, int, str]:
    """
    Parse scene_number, shot_number, version from an s3_key filename.
    e.g. phase3/.../scene_2_shot_3_v1.mp4 → (2, 3, "v1")
    """
    filename = s3_key.rsplit("/", 1)[-1]
    m = re.search(r"scene_(\d+)_shot_(\d+)_v(\d+)", filename, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), f"v{m.group(3)}"
    return 0, 0, "v1"


def _build_clip_ref(
    s3_key: str,
    s3_url: str,
    attempt_key: str,
    approval_status: str,
    duration: float,
    has_audio: bool,
    shot_id: str,
    scene_number: int,
    shot_number: int,
    version: str,
    description: str,
) -> ClipRef:
    return ClipRef(
        shot_id=shot_id,
        scene_number=scene_number,
        shot_number=shot_number,
        version=version,
        attempt_key=attempt_key,
        s3_key=s3_key,
        s3_url=s3_url,
        filename=s3_key.rsplit("/", 1)[-1],
        description=description,
        duration=duration,
        has_audio=has_audio,
        approval_status=approval_status,
    )


# ---------------------------------------------------------------------------
# initialize_node
# ---------------------------------------------------------------------------

def initialize_node(state: Phase4State) -> Phase4State:
    """
    Entry node for the Phase 4 graph.

    Reads Agent 0's human video selections from MongoDB, resolves each shot
    into a ShotCandidates group (with fresh presigned URLs + ffprobe metadata),
    loads the script/shotlist/title, sets all counters to zero, upserts the
    final_assemblies document, and marks the pipeline as running.
    """
    show_id = state.get("show_id")
    episode_number = state.get("episode_number")
    episode_id = state.get("episode_id", "")
    movie_id = state.get("movie_id")
    job_id = state.get("job_id")

    logger.info(
        f"[initialize_node] Starting Phase 4: show_id={show_id}, "
        f"episode={episode_number}, episode_id={episode_id}, job_id={job_id}"
    )

    os.makedirs(_TMP_DIR, exist_ok=True)

    mongo_client = None
    s3_client = None
    tmp_files: List[str] = []

    try:
        # ── clients ──────────────────────────────────────────────────────────
        mongo_uri = os.environ["MONGODB_ATLAS_URI"]
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client.get_database(
            os.environ.get("production_MONGODB_DATABASE_NAME", "production")
        )

        bucket_name = os.environ.get("production_S3_BUCKET_NAME", "zeroshot-v1")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("production_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("production_AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("production_AWS_REGION", "eu-north-1"),
        )

        # ── 1. Load shot documents ────────────────────────────────────────────
        # Master movie runs store one project/show id per scene. When movie_id is
        # available, load every scene project so Phase 4 assembles the full movie.
        query_show_ids = [show_id]
        is_movie_run = False
        if movie_id:
            try:
                movie_doc = db[_MOVIES_COLLECTION].find_one({"_id": ObjectId(movie_id)})
                project_ids = [str(pid) for pid in (movie_doc or {}).get("project_ids", [])]
                if project_ids:
                    query_show_ids = project_ids
                    is_movie_run = True
            except Exception as e:
                logger.warning(f"[initialize_node] Could not resolve movie project_ids: {e}")

        query: Dict[str, Any] = (
            {"show_id": {"$in": query_show_ids}}
            if len(query_show_ids) > 1
            else {"show_id": query_show_ids[0]}
        )
        if episode_number is not None and not is_movie_run:
            query["episode_number"] = episode_number

        shot_docs = list(db[_SHOTS_COLLECTION].find(query).sort("episode_number", 1))
        logger.info(
            f"[initialize_node] Found {len(shot_docs)} shot document(s) "
            f"for show_ids={query_show_ids}"
        )

        # ── 2. Build clip_manifest ────────────────────────────────────────────
        clip_manifest: List[ShotCandidates] = []

        for doc in shot_docs:
            raw_shots = doc.get("annotated_shots") or doc.get("shots") or []

            for shot in raw_shots:
                shot_id: str = shot.get("shot_id", "")
                description: str = shot.get("description", "")
                shot_scene: int = int(shot.get("scene_number") or 0)
                shot_number: int = int(shot.get("sequence_number") or shot.get("shot_number") or 0)

                video_obj: Dict = shot.get("video") or {}
                if not isinstance(video_obj, dict):
                    continue

                # Collect all available versions
                all_versions: List[Dict] = []
                for attempt_key, v in sorted(video_obj.items()):
                    if not (attempt_key.startswith("v") and attempt_key[1:].isdigit()):
                        continue
                    if not isinstance(v, dict):
                        continue
                    urls = v.get("generated_videos_s3") or []
                    if not urls:
                        continue
                    raw_url = urls[0]
                    s3_key = _parse_s3_key(raw_url, bucket_name)
                    all_versions.append({
                        "attempt_key": attempt_key,
                        "s3_key": s3_key,
                        "approval_status": v.get("approval_status", "pending"),
                    })

                if not all_versions:
                    logger.warning(f"[initialize_node] No video versions found for shot {shot_id} — skipping.")
                    continue

                # Determine candidate set from Agent 0 selection
                sel_doc = shot.get("video_review_selection") or {}
                selected_list: List[Dict] = sel_doc.get("selected", [])

                if len(selected_list) == 1:
                    selection_mode = "single"
                    selection_source = "human"
                    candidate_keys = [selected_list[0]["s3_key"]]
                elif len(selected_list) >= 2:
                    selection_mode = "choose_one"
                    selection_source = "human"
                    candidate_keys = [s["s3_key"] for s in selected_list]
                else:
                    # No selection → fallback: all versions
                    selection_mode = "choose_one"
                    selection_source = "fallback_all_versions"
                    candidate_keys = [v["s3_key"] for v in all_versions]

                # Build ClipRef for each candidate
                candidates: List[ClipRef] = []
                for s3_key in candidate_keys:
                    # Find the matching version metadata
                    version_meta = next(
                        (v for v in all_versions if v["s3_key"] == s3_key),
                        {"attempt_key": "v0", "approval_status": "pending"},
                    )
                    attempt_key = version_meta["attempt_key"]
                    approval_status = version_meta["approval_status"]

                    # Re-mint fresh presigned URL
                    try:
                        fresh_url = _mint_presigned_url(s3_client, bucket_name, s3_key)
                    except Exception as e:
                        logger.warning(f"[initialize_node] Could not mint URL for {s3_key}: {e}")
                        continue

                    # Download + probe
                    local_path = os.path.join(
                        _TMP_DIR, f"init_{shot_id}_{attempt_key}_{os.path.basename(s3_key)}"
                    )
                    tmp_files.append(local_path)
                    try:
                        _download_clip(fresh_url, local_path)
                        duration, has_audio = _ffprobe_clip(local_path)
                    except Exception as e:
                        logger.warning(f"[initialize_node] Download/probe failed for {s3_key}: {e}")
                        duration, has_audio = 0.0, False

                    # Parse scene/shot/version from filename
                    scene_n, shot_n, file_ver = _parse_scene_shot_version(s3_key)
                    scene_n = scene_n or shot_scene
                    shot_n = shot_n or shot_number

                    candidates.append(_build_clip_ref(
                        s3_key=s3_key,
                        s3_url=fresh_url,
                        attempt_key=attempt_key,
                        approval_status=approval_status,
                        duration=duration,
                        has_audio=has_audio,
                        shot_id=shot_id or f"scene_{scene_n}_shot_{shot_n}",
                        scene_number=scene_n,
                        shot_number=shot_n,
                        version=file_ver,
                        description=description,
                    ))

                if not candidates:
                    logger.warning(f"[initialize_node] No candidates resolved for shot {shot_id} — skipping.")
                    continue

                clip_manifest.append(ShotCandidates(
                    shot_id=shot_id or f"scene_{candidates[0]['scene_number']}_shot_{candidates[0]['shot_number']}",
                    scene_number=candidates[0]["scene_number"],
                    shot_number=candidates[0]["shot_number"],
                    selection_mode=selection_mode,
                    selection_source=selection_source,
                    candidates=candidates,
                ))

        if not clip_manifest:
            raise ValueError(
                f"[initialize_node] Zero candidates resolved for show_id={show_id}, "
                f"episode={episode_number}. "
                "Check that Phase 3 videos exist and Agent 0 selections are saved."
            )

        # Sort manifest by (scene_number, shot_number)
        clip_manifest.sort(key=lambda g: (g["scene_number"], g["shot_number"]))
        logger.info(f"[initialize_node] clip_manifest: {len(clip_manifest)} shot group(s)")

        # ── 3. Load title, script_content, shot_list ──────────────────────────
        title = state.get("title", "")
        if not title:
            # Try movies collection
            try:
                movie_doc = db[_MOVIES_COLLECTION].find_one(
                    {"project_ids": show_id} if movie_id is None
                    else {"_id": ObjectId(movie_id)}
                )
                if not movie_doc:
                    movie_doc = db[_MOVIES_COLLECTION].find_one({"show_id": show_id})
                if movie_doc:
                    title = movie_doc.get("title") or movie_doc.get("name") or ""
            except Exception as e:
                logger.warning(f"[initialize_node] Could not load title from movies: {e}")

        title = title or f"Episode_{episode_number}"
        title_safe = re.sub(r"[^A-Za-z0-9_]", "_", title.upper()).strip("_")

        script_content = state.get("script_content", "")
        shot_list = state.get("shot_list", {})

        if not script_content or not shot_list:
            # Try to find script/shotlist in shot docs or a scripts collection
            for doc in shot_docs:
                if not script_content:
                    script_content = (
                        doc.get("script_content")
                        or doc.get("script")
                        or doc.get("episode_script")
                        or ""
                    )
                if not shot_list:
                    shot_list = (
                        doc.get("shot_list")
                        or doc.get("shots_overview")
                        or {}
                    )
                if script_content and shot_list:
                    break

        # ── 4. Set defaults ───────────────────────────────────────────────────
        aspect_ratio = (
            state.get("aspect_ratio")
            or os.environ.get("PHASE4_TARGET_ASPECT_RATIO", "9:16")
        )
        raw_platforms = os.environ.get("PHASE4_TARGET_PLATFORMS", "reels,tiktok,shorts")
        target_platforms = state.get("target_platforms") or [
            p.strip() for p in raw_platforms.split(",") if p.strip()
        ]

        state["clip_manifest"] = clip_manifest
        state["title"] = title
        state["script_content"] = script_content
        state["shot_list"] = shot_list
        state["aspect_ratio"] = aspect_ratio
        state["target_platforms"] = target_platforms

        # Zero all counters / version numbers
        state["edl_version"] = 0
        state["rough_cut_version"] = 0
        state["edit_loop_count"] = 0
        state["vo_version"] = 0
        state["music_version"] = 0
        state["vo_preview_version"] = 0
        state["final_master_version"] = 0
        state.setdefault("errors", [])
        state["pipeline_status"] = "running"
        state["current_agent"] = "initialize"

        logger.info(
            f"[initialize_node] title={title!r}, title_safe={title_safe!r}, "
            f"aspect_ratio={aspect_ratio}, platforms={target_platforms}"
        )

        # ── 5. Upsert final_assemblies document ───────────────────────────────
        try:
            upsert_assembly(
                db=db,
                show_id=show_id,
                episode_number=episode_number,
                episode_id=episode_id,
                movie_id=movie_id,
                title=title,
                clip_manifest=[dict(g) for g in clip_manifest],
            )
        except Exception as e:
            logger.warning(f"[initialize_node] upsert_assembly failed: {e}")

        # ── 6. Mark pipeline running in production_pipelines ──────────────────
        if job_id:
            try:
                ps = PipelineService()
                ps.update_job_status(job_id=job_id, agent_number=0, status="running")
                ps.update_job_current_agent(job_id=job_id, current_agent="initialize")
            except Exception as e:
                logger.warning(f"[initialize_node] Could not update job status: {e}")

        logger.info("[initialize_node] Complete — handing off to agent_1_edl_node.")
        return state

    except Exception as e:
        logger.error(f"[initialize_node] Failed: {e}", exc_info=True)
        state.setdefault("errors", []).append({"agent": "initialize", "error": str(e)})
        state["pipeline_status"] = "failed"
        raise

    finally:
        for path in tmp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        if mongo_client:
            mongo_client.close()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_phase4_graph():
    """
    Construct and compile the Phase 4 StateGraph.

    Node naming mirrors the function names so LangGraph's introspection output
    is readable.

    Returns:
        CompiledStateGraph ready to invoke with .invoke(initial_state).
    """
    graph = StateGraph(Phase4State)

    # ── nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("initialize_node", initialize_node)
    graph.add_node("agent_1_edl_node", agent_1_edl_node)
    graph.add_node("agent_2_assembly_node", agent_2_assembly_node)
    graph.add_node("agent_3_review_node", agent_3_review_node)
    graph.add_node("agent_4_revise_node", agent_4_revise_node)
    graph.add_node("agent_5a_director_node", agent_5a_director_node)
    graph.add_node("agent_5b_tts_node", agent_5b_tts_node)
    graph.add_node("agent_6_av_merge_node", agent_6_av_merge_node)
    graph.add_node("agent_7a_music_director_node", agent_7a_music_director_node)
    graph.add_node("agent_7b_lyria_node", agent_7b_lyria_node)
    graph.add_node("agent_8_final_mix_node", agent_8_final_mix_node)
    graph.add_node("agent_9_delivery_node", agent_9_delivery_node)

    # ── entry ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("initialize_node")

    # ── linear edges ──────────────────────────────────────────────────────────
    graph.add_edge("initialize_node", "agent_1_edl_node")
    graph.add_edge("agent_1_edl_node", "agent_2_assembly_node")
    graph.add_edge("agent_2_assembly_node", "agent_3_review_node")

    # ── conditional: review → revise loop OR advance to VO ────────────────────
    # route_after_review returns "revise" (≤2 loops) or "vo" (approved / cap hit)
    graph.add_conditional_edges(
        "agent_3_review_node",
        route_after_review,
        {
            "revise": "agent_4_revise_node",
            "vo": "agent_5a_director_node",
        },
    )

    # Agent 4 → back to assembly (edit_loop_count cap enforced inside route_after_review)
    graph.add_edge("agent_4_revise_node", "agent_2_assembly_node")

    # ── VO: 5A Director → 5B TTS ──────────────────────────────────────────────
    graph.add_edge("agent_5a_director_node", "agent_5b_tts_node")
    graph.add_edge("agent_5b_tts_node", "agent_6_av_merge_node")

    # ── AV merge → Music: 7A Director → 7B Lyria ──────────────────────────────
    graph.add_edge("agent_6_av_merge_node", "agent_7a_music_director_node")
    graph.add_edge("agent_7a_music_director_node", "agent_7b_lyria_node")
    graph.add_edge("agent_7b_lyria_node", "agent_8_final_mix_node")

    # ── conditional: final mix → delivery or END ──────────────────────────────
    graph.add_conditional_edges(
        "agent_8_final_mix_node",
        route_after_mix,
        {
            "deliver": "agent_9_delivery_node",
            "end": END,
        },
    )

    graph.add_edge("agent_9_delivery_node", END)

    return graph.compile()


# Module-level compiled graph (lazy singleton)
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_phase4_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public invocation entry point
# ---------------------------------------------------------------------------

def run_phase4_pipeline(
    show_id: str,
    episode_number: int,
    episode_id: str,
    *,
    movie_id: Optional[str] = None,
    project_id: Optional[str] = None,
    job_id: Optional[str] = None,
    title: Optional[str] = None,
    script_content: Optional[str] = None,
    shot_list: Optional[Dict] = None,
    aspect_ratio: Optional[str] = None,
    target_platforms: Optional[List[str]] = None,
) -> Phase4State:
    """
    Seed an initial Phase4State and run the full Phase 4 pipeline synchronously.

    Args:
        show_id:          Project / show identifier (maps to Phase 3 show_id).
        episode_number:   Episode number (int).
        episode_id:       Episode string ID (e.g. "S01E01").
        movie_id:         Optional MongoDB ObjectId string for the movies document.
        project_id:       Optional project identifier alias.
        job_id:           Optional production_pipelines job ObjectId for status tracking.
        title:            Optional title override (initialize_node loads from DB if absent).
        script_content:   Optional pre-loaded script text.
        shot_list:        Optional pre-loaded shot list dict.
        aspect_ratio:     Output aspect ratio override (default: PHASE4_TARGET_ASPECT_RATIO env).
        target_platforms: Platform list override (default: PHASE4_TARGET_PLATFORMS env).

    Returns:
        The final Phase4State after the graph completes.

    Raises:
        Exception: Re-raises any node exception after recording it in the state.
    """
    initial_state: Dict[str, Any] = {
        "show_id": show_id,
        "episode_number": episode_number,
        "episode_id": episode_id,
        "movie_id": movie_id,
        "project_id": project_id,
        "job_id": job_id,
        "title": title or "",
        "script_content": script_content or "",
        "shot_list": shot_list or {},
        "aspect_ratio": aspect_ratio or "",
        "target_platforms": target_platforms or [],
        "clip_manifest": [],
        "errors": [],
        "pipeline_status": "pending",
        "current_agent": "pending",
        # Version counters — initialize_node will set these to 0
        "edl_version": 0,
        "rough_cut_version": 0,
        "edit_loop_count": 0,
        "vo_version": 0,
        "music_version": 0,
        "vo_preview_version": 0,
        "final_master_version": 0,
    }

    logger.info(
        f"[run_phase4_pipeline] Starting: show_id={show_id}, "
        f"episode={episode_number}, job_id={job_id}"
    )

    try:
        compiled = _get_graph()
        final_state = compiled.invoke(initial_state)
        logger.info(
            f"[run_phase4_pipeline] Completed: "
            f"pipeline_status={final_state.get('pipeline_status')}, "
            f"final_master={final_state.get('final_master_s3_key', 'n/a')}"
        )
        return final_state

    except Exception as e:
        logger.error(f"[run_phase4_pipeline] Pipeline failed: {e}", exc_info=True)
        # Best-effort failure recording
        try:
            mongo_client = MongoClient(os.environ.get("MONGODB_ATLAS_URI", ""))
            db = mongo_client.get_database(
                os.environ.get("production_MONGODB_DATABASE_NAME", "production")
            )
            db["final_assemblies"].update_one(
                {"show_id": show_id, "episode_number": episode_number},
                {"$set": {
                    "pipeline_status": "failed",
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
            if job_id:
                ps = PipelineService()
                ps.update_job_status(job_id=job_id, agent_number=0, status="failed")
            mongo_client.close()
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("Building Phase 4 graph...")
    g = build_phase4_graph()
    print("\nNodes:")
    for node in g.nodes:
        print(f"  {node}")
    print("\nEdges:")
    for src, targets in g.edges.items():
        for tgt in (targets if isinstance(targets, list) else [targets]):
            print(f"  {src} → {tgt}")
    print("\nPhase 4 graph OK.")
    sys.exit(0)
