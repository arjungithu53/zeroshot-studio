"""
final_assemblies_service — MongoDB service layer for the `final_assemblies` collection.

Phase 4 equivalent of assets_collection_service / project_service. Every Phase 4
agent calls these functions rather than duplicating pymongo upsert logic.

Celery fork model: the caller opens a fresh MongoClient and passes the resulting
Database object as `db`. This module never opens connections itself.

Collection: `final_assemblies`
Document key: { show_id, episode_number, episode_id }
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "upsert_assembly",
    "update_agent_output",
    "append_versioned",
    "set_deliverables",
    "set_pipeline_status",
    "get_assembly",
]

_COLLECTION = "final_assemblies"

_VERSIONED_ARRAYS = frozenset({
    "edl_versions",
    "rough_cuts",
    "reviews",
    "vo",
    "vo_preview",
    "music",
    "final_masters",
})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _col(db: Any):
    """Return the final_assemblies collection from a pymongo Database."""
    return db[_COLLECTION]


def _key(show_id: str, episode_number: int, episode_id: str) -> Dict:
    """Canonical filter for a final_assemblies document."""
    return {
        "show_id": show_id,
        "episode_number": episode_number,
        "episode_id": episode_id,
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _require(result, show_id: str, episode_number: int, fn: str) -> None:
    """Raise ValueError if the update matched nothing."""
    if result.matched_count == 0:
        raise ValueError(
            f"[{fn}] final_assemblies document not found for "
            f"show_id={show_id!r}, episode_number={episode_number}. "
            "Call upsert_assembly first."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_assembly(
    db: Any,
    show_id: str,
    episode_number: int,
    episode_id: str,
    movie_id: Optional[str] = None,
    title: Optional[str] = None,
    clip_manifest: Optional[List[Dict]] = None,
) -> None:
    """
    Create the final_assemblies document if it doesn't exist; otherwise refresh
    pipeline_status and updated_at.  Safe to call repeatedly.

    On insert:  initialises all versioned arrays to [], agent_outputs to {},
                deliverables to {}, and records created_at.
    On update:  only pipeline_status, updated_at (and optional fields when
                supplied) are overwritten — existing versioned data is preserved.
    """
    now = _now()
    col = _col(db)

    set_fields: Dict[str, Any] = {
        "pipeline_status": "running",
        "updated_at": now,
    }
    if movie_id is not None:
        set_fields["movie_id"] = movie_id
    if title is not None:
        set_fields["title"] = title
    if clip_manifest is not None:
        set_fields["clip_manifest"] = clip_manifest

    set_on_insert: Dict[str, Any] = {
        "show_id": show_id,
        "episode_number": episode_number,
        "episode_id": episode_id,
        "agent_outputs": {},
        "deliverables": {},
        "created_at": now,
    }
    for array_name in _VERSIONED_ARRAYS:
        set_on_insert[array_name] = []

    col.update_one(
        filter=_key(show_id, episode_number, episode_id),
        update={"$set": set_fields, "$setOnInsert": set_on_insert},
        upsert=True,
    )
    logger.info(
        f"[upsert_assembly] show_id={show_id} episode={episode_number} "
        f"episode_id={episode_id} — upserted"
    )


def update_agent_output(
    db: Any,
    show_id: str,
    episode_number: int,
    agent_number: int,
    status: str,
    output: Dict[str, Any],
) -> None:
    """
    Set agent_outputs.agent{N}.status / .executed_at / .output and updated_at.

    Mirrors the Phase 1–3 agent_outputs.agent{N} pattern exactly.

    Args:
        db:             pymongo Database (passed by the Celery node).
        show_id:        Show identifier.
        episode_number: Episode number.
        agent_number:   Agent index (0 for initialize, 1–9 for agents).
        status:         "running" | "completed" | "failed" | "skipped" | "retrying".
        output:         Serialisable dict of the agent's result payload.

    Raises:
        ValueError: If the document does not exist.
    """
    prefix = f"agent_outputs.agent{agent_number}"
    result = _col(db).update_one(
        filter=_key(show_id, episode_number, ""),
        update={"$set": {
            f"{prefix}.status":      status,
            f"{prefix}.executed_at": _now(),
            f"{prefix}.output":      output,
            "updated_at":            _now(),
        }},
    )
    # episode_id may differ; retry with a broader filter if first miss
    if result.matched_count == 0:
        result = _col(db).update_one(
            filter={"show_id": show_id, "episode_number": episode_number},
            update={"$set": {
                f"{prefix}.status":      status,
                f"{prefix}.executed_at": _now(),
                f"{prefix}.output":      output,
                "updated_at":            _now(),
            }},
        )
    _require(result, show_id, episode_number, "update_agent_output")
    logger.info(
        f"[update_agent_output] agent{agent_number} → {status} "
        f"show_id={show_id} episode={episode_number}"
    )


def append_versioned(
    db: Any,
    show_id: str,
    episode_number: int,
    array_field: str,
    entry: Dict[str, Any],
) -> None:
    """
    Push `entry` onto a versioned array (edl_versions, rough_cuts, reviews,
    vo, vo_preview, music, final_masters) and bump updated_at.

    Args:
        db:             pymongo Database.
        show_id:        Show identifier.
        episode_number: Episode number.
        array_field:    Name of the array field (must be one of _VERSIONED_ARRAYS).
        entry:          Dict to push onto the array.

    Raises:
        ValueError: If array_field is not a recognised versioned array, or if
                    the parent document does not exist.
    """
    if array_field not in _VERSIONED_ARRAYS:
        raise ValueError(
            f"[append_versioned] Unknown array_field {array_field!r}. "
            f"Must be one of: {sorted(_VERSIONED_ARRAYS)}"
        )
    result = _col(db).update_one(
        filter={"show_id": show_id, "episode_number": episode_number},
        update={
            "$push": {array_field: entry},
            "$set":  {"updated_at": _now()},
        },
    )
    _require(result, show_id, episode_number, "append_versioned")
    logger.info(
        f"[append_versioned] {array_field} ← entry appended "
        f"show_id={show_id} episode={episode_number}"
    )


def set_deliverables(
    db: Any,
    show_id: str,
    episode_number: int,
    deliverables: Dict[str, Any],
) -> None:
    """
    Set final_assemblies.deliverables and bump updated_at.

    Args:
        db:             pymongo Database.
        show_id:        Show identifier.
        episode_number: Episode number.
        deliverables:   Dict with keys like captions_srt, exports, final_metadata.

    Raises:
        ValueError: If the document does not exist.
    """
    result = _col(db).update_one(
        filter={"show_id": show_id, "episode_number": episode_number},
        update={"$set": {
            "deliverables": deliverables,
            "updated_at":   _now(),
        }},
    )
    _require(result, show_id, episode_number, "set_deliverables")
    logger.info(
        f"[set_deliverables] deliverables written "
        f"show_id={show_id} episode={episode_number}"
    )


def set_pipeline_status(
    db: Any,
    show_id: str,
    episode_number: int,
    status: str,
) -> None:
    """
    Set pipeline_status and bump updated_at.

    Args:
        db:             pymongo Database.
        show_id:        Show identifier.
        episode_number: Episode number.
        status:         "running" | "completed" | "failed".

    Raises:
        ValueError: If the document does not exist.
    """
    result = _col(db).update_one(
        filter={"show_id": show_id, "episode_number": episode_number},
        update={"$set": {
            "pipeline_status": status,
            "updated_at":      _now(),
        }},
    )
    _require(result, show_id, episode_number, "set_pipeline_status")
    logger.info(
        f"[set_pipeline_status] → {status!r} "
        f"show_id={show_id} episode={episode_number}"
    )


def get_assembly(
    db: Any,
    show_id: str,
    episode_number: int,
    episode_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the full final_assemblies document or None if not found.

    Args:
        db:             pymongo Database.
        show_id:        Show identifier.
        episode_number: Episode number.
        episode_id:     Episode identifier (e.g. "S01E01").

    Returns:
        Document dict with _id serialised to str, or None.
    """
    doc = _col(db).find_one(_key(show_id, episode_number, episode_id))
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc
