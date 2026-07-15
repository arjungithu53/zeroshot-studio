"""
Shot Video Review — Streamlit app for human checkpoint before Phase 4.

Run with:
    streamlit run tools/video_review.py --server.port 8004
"""

import streamlit as st
import requests

st.set_page_config(page_title="Shot Video Review", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers(admin_key: str) -> dict:
    return {"X-Admin-Key": admin_key} if admin_key else {}


def _parse_int(val: str) -> int | None:
    try:
        return int(val.strip()) if val.strip() else None
    except ValueError:
        return None


def fetch_video_bytes(url: str, s3_key: str, vid_cache: dict) -> bytes | None:
    """Download video server-side so private S3 presigned URLs work in the browser.
    Cache key is s3_key (not URL) because presigned URLs expire between reruns."""
    if s3_key in vid_cache:
        return vid_cache[s3_key]
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.content
        vid_cache[s3_key] = data
        return data
    except Exception:
        vid_cache[s3_key] = None
        return None


def fetch_gallery(api_url: str, admin_key: str, movie_id: str,
                  scene_number: int | None) -> dict | None:
    url = f"{api_url.rstrip('/')}/video-review/{movie_id}"
    params = {}
    if scene_number is not None:
        params["scene_number"] = scene_number
    try:
        resp = requests.get(url, headers=_headers(admin_key), params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("data")
    except requests.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Failed to load gallery: {e}")
    return None


def save_selection(api_url: str, admin_key: str, movie_id: str,
                   shot_id: str, selected_list: list) -> bool:
    endpoint = f"{api_url.rstrip('/')}/video-review/{movie_id}/{shot_id}/select"
    try:
        resp = requests.post(
            endpoint,
            json={"selected": selected_list},
            headers=_headers(admin_key),
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Save failed for {shot_id}: {e}")
        return False


def continue_to_phase4(api_url: str, admin_key: str, master_job_id: str) -> bool:
    """Signal the master pipeline to proceed from video review into Phase 4."""
    endpoint = f"{api_url.rstrip('/')}/master/continue-to-phase4/{master_job_id}"
    try:
        resp = requests.post(endpoint, headers=_headers(admin_key), timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to continue to Phase 4: {e}")
        return False


# ---------------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.get("api_url", "http://localhost:8000/api/v1/phase4"),
    )
    admin_key = st.text_input(
        "Admin Key (optional)", type="password", value=st.session_state.get("admin_key", "")
    )
    movie_id = st.text_input("Movie ID", value=st.session_state.get("movie_id", ""))
    master_job_id = st.text_input(
        "Master Job ID (optional)",
        value=st.session_state.get("master_job_id", ""),
        placeholder="Fill to auto-continue to Phase 4 after saving",
    )

    st.markdown("**Scene filter** (leave blank to load all scenes)")
    scene_input = st.text_input(
        "Scene Number",
        value=st.session_state.get("scene_input", ""),
        placeholder="e.g. 2  —  blank = all scenes",
    )

    st.session_state.update(
        api_url=api_url,
        admin_key=admin_key,
        movie_id=movie_id,
        master_job_id=master_job_id,
        scene_input=scene_input,
    )

    load_clicked = st.button("Load Gallery", type="primary")
    if load_clicked:
        for k in ("gallery", "selections", "vid_cache", "_cleared_shots"):
            st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# Trigger load
# ---------------------------------------------------------------------------
if load_clicked:
    if not movie_id:
        st.warning("Please enter a Movie ID.")
    else:
        scene_filter = _parse_int(scene_input)
        with st.spinner("Loading gallery…"):
            data = fetch_gallery(api_url, admin_key, movie_id, scene_filter)
            if data is not None:
                st.session_state["gallery"] = data
                sels: dict = {}
                for scene in data.get("scenes", []):
                    for shot in scene.get("shots", []):
                        sid = shot["shot_id"]
                        existing = shot.get("current_selection", [])
                        if existing:
                            sel_versions = [
                                v for v in shot.get("versions", [])
                                if v["version"] in existing
                            ]
                            if sel_versions:
                                sels[sid] = sel_versions
                st.session_state["selections"] = sels
                st.session_state["vid_cache"] = {}
                st.session_state["_cleared_shots"] = set()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("Shot Video Review")

gallery = st.session_state.get("gallery")
if not gallery:
    st.info("Enter your Movie ID in the sidebar and click **Load Gallery**.")
    st.stop()

all_shots = [
    shot
    for scene in gallery.get("scenes", [])
    for shot in scene.get("shots", [])
]
total = len(all_shots)

if total == 0:
    filter_msg = f" (scene filter: {scene_input.strip()})" if scene_input.strip() else ""
    st.warning(
        f"No shots found for movie `{movie_id}`{filter_msg}. "
        "Check the Movie ID or try a different scene filter."
    )
    st.stop()

selections: dict = st.session_state.setdefault("selections", {})
vid_cache: dict = st.session_state.setdefault("vid_cache", {})
cleared_shots: set = st.session_state.setdefault("_cleared_shots", set())

reviewed = sum(1 for s in all_shots if selections.get(s["shot_id"]))
pct = reviewed / total if total else 0

col_prog, col_filter = st.columns([4, 1])
with col_prog:
    st.progress(pct, text=f"{reviewed} / {total} shots reviewed  ({int(pct * 100)}%)")
with col_filter:
    hide_reviewed = st.checkbox("Hide reviewed shots")

st.divider()

# ---------------------------------------------------------------------------
# Scene / Shot gallery
# ---------------------------------------------------------------------------
for scene in sorted(gallery.get("scenes", []), key=lambda s: s.get("scene_number") or 0):
    scene_num = scene.get("scene_number")
    scene_label = f"Scene {scene_num}" if scene_num is not None else "Unassigned Scene"
    st.subheader(scene_label)

    for shot in scene.get("shots", []):
        shot_id = shot["shot_id"]
        versions = shot.get("versions", [])
        current_sel = shot.get("current_selection", [])
        shot_selections = selections.get(shot_id, [])
        already_selected = bool(shot_selections or current_sel)

        if hide_reviewed and already_selected:
            continue

        shot_num = shot.get("shot_number", "?")
        label = ("✅ " if already_selected else "") + f"Shot {shot_num}  —  {shot_id}"

        with st.expander(label, expanded=not already_selected):
            desc = shot.get("description", "")
            if desc:
                st.caption(desc)

            if shot_selections:
                labels = ", ".join(v["version"] for v in shot_selections)
                st.success(f"Selected: **{labels}**")
            elif current_sel:
                st.info(f"Saved selection: {', '.join(current_sel)}")

            if not versions:
                st.warning("No video versions found for this shot.")
                continue

            COLS = 3
            selected_keys = {v["s3_key"] for v in shot_selections}

            for chunk_start in range(0, len(versions), COLS):
                chunk = versions[chunk_start : chunk_start + COLS]
                cols = st.columns(len(chunk))
                for col, ver in zip(cols, chunk):
                    with col:
                        s3_key = ver["s3_key"]
                        s3_url = ver["s3_url"]

                        if s3_key not in vid_cache:
                            with st.spinner(f"Loading {ver['version']}…"):
                                fetch_video_bytes(s3_url, s3_key, vid_cache)
                                st.session_state["vid_cache"] = vid_cache

                        vid_bytes = vid_cache.get(s3_key)
                        if vid_bytes:
                            st.video(vid_bytes)
                        else:
                            try:
                                st.video(s3_url)
                            except Exception:
                                st.error(f"Could not load {ver['version']}")

                        st.caption(
                            f"**{ver['version']}** (attempt {ver['attempt_key']})  \n"
                            f"Status: {ver.get('approval_status', 'unknown')}"
                        )

                        is_checked = s3_key in selected_keys
                        checked = st.checkbox(
                            "Use this version",
                            value=is_checked,
                            key=f"chk_{shot_id}_{ver['version']}_{ver['attempt_key']}",
                        )

                        if checked != is_checked:
                            current_list = list(selections.get(shot_id, []))
                            if checked:
                                current_list.append({
                                    "version":     ver["version"],
                                    "attempt_key": ver["attempt_key"],
                                    "s3_key":      ver["s3_key"],
                                    "s3_url":      ver["s3_url"],
                                })
                            else:
                                current_list = [v for v in current_list if v["s3_key"] != s3_key]
                            selections[shot_id] = current_list
                            st.session_state["selections"] = selections
                            if not current_list:
                                cleared_shots.add(shot_id)
                            else:
                                cleared_shots.discard(shot_id)
                            st.session_state["_cleared_shots"] = cleared_shots
                            st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------
col_save, col_reset = st.columns([2, 1])
with col_save:
    if st.button("Save All Selections", type="primary"):
        all_shot_ids = set(selections.keys()) | cleared_shots
        saved = failed = 0
        with st.spinner("Saving…"):
            for sid in all_shot_ids:
                sel_list = selections.get(sid, [])
                ok = save_selection(api_url, admin_key, movie_id, sid, sel_list)
                if ok:
                    saved += 1
                else:
                    failed += 1
        if saved:
            st.success(f"Saved {saved} shot selection(s).")
        if failed:
            st.error(f"{failed} shot selection(s) failed to save.")
        if saved and master_job_id.strip():
            with st.spinner("Signalling pipeline to continue to Phase 4…"):
                ok = continue_to_phase4(api_url, admin_key, master_job_id.strip())
            if ok:
                st.success("Pipeline resumed — Phase 4 is now running.")
        if saved:
            data = fetch_gallery(api_url, admin_key, movie_id, _parse_int(scene_input))
            if data:
                st.session_state["gallery"] = data
            st.rerun()

with col_reset:
    if st.button("Reset Selections"):
        st.session_state["selections"] = {}
        st.session_state["_cleared_shots"] = set()
        st.rerun()
