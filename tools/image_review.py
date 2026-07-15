"""
Shot Image Review — Streamlit app for human checkpoint before Phase 3.

Run with:
    streamlit run tools/image_review.py
"""

import io
import streamlit as st
import requests

st.set_page_config(page_title="Shot Image Review", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers(admin_key: str) -> dict:
    return {"X-Admin-Key": admin_key} if admin_key else {}


def fetch_image_bytes(url: str) -> bytes | None:
    """Download image server-side so private S3 URLs work in the browser."""
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def fetch_gallery(api_url: str, admin_key: str, movie_id: str,
                  scene_number: int | None) -> dict | None:
    url = f"{api_url.rstrip('/')}/image-review/{movie_id}"
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


def continue_to_phase3(api_url: str, admin_key: str, master_job_id: str) -> bool:
    """Signal the master pipeline to proceed from image review into Phase 3."""
    # api_url is like http://localhost:8000/api/v1/phase2 — replace last segment with master
    parts = api_url.rstrip("/").split("/")
    parts[-1] = "master"
    endpoint = "/".join(parts) + f"/continue-to-phase3/{master_job_id}"
    try:
        resp = requests.post(endpoint, headers=_headers(admin_key), timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to continue to Phase 3: {e}")
        return False


def save_selection(api_url: str, admin_key: str, movie_id: str,
                   shot_id: str, version: str, index: int, url: str) -> bool:
    endpoint = f"{api_url.rstrip('/')}/image-review/{movie_id}/{shot_id}/select"
    try:
        resp = requests.post(
            endpoint,
            json={"version": version, "index": index, "url": url},
            headers=_headers(admin_key),
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Save failed for {shot_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    api_url      = st.text_input("API Base URL", value=st.session_state.get("api_url", "http://localhost:8000/api/v1/phase2"))
    admin_key    = st.text_input("Admin Key (optional)", type="password", value=st.session_state.get("admin_key", ""))
    movie_id     = st.text_input("Movie ID", value=st.session_state.get("movie_id", ""))
    master_job_id = st.text_input(
        "Master Job ID (optional)",
        value=st.session_state.get("master_job_id", ""),
        placeholder="Fill to auto-continue to Phase 3 after saving",
    )

    st.markdown("**Scene filter** (leave blank to load all scenes)")
    scene_input = st.text_input("Scene Number", value=st.session_state.get("scene_input", ""),
                                placeholder="e.g. 2  —  blank = all scenes")

    st.session_state.update(api_url=api_url, admin_key=admin_key, movie_id=movie_id,
                            master_job_id=master_job_id, scene_input=scene_input)

    load_clicked = st.button("Load Shots", type="primary")
    if load_clicked:
        for k in ("gallery", "selections", "img_cache"):
            st.session_state.pop(k, None)


def _parse_int(val: str) -> int | None:
    try:
        return int(val.strip()) if val.strip() else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Trigger load
# ---------------------------------------------------------------------------
if load_clicked:
    if not movie_id:
        st.warning("Please enter a Movie ID.")
    else:
        scene_filter = _parse_int(scene_input)
        with st.spinner("Loading shots…"):
            data = fetch_gallery(api_url, admin_key, movie_id, scene_filter)
            if data is not None:
                st.session_state["gallery"] = data
                sels = {}
                for s in data.get("shots", []):
                    sel = s.get("selected")
                    if sel:
                        sels[s["shot_id"]] = {"version": sel["version"],
                                               "index": sel["index"], "url": sel["url"]}
                st.session_state["selections"] = sels
                st.session_state["img_cache"]  = {}

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("Shot Image Review")

gallery = st.session_state.get("gallery")
if not gallery:
    st.info("Enter your Movie ID in the sidebar and click **Load Shots**.")
    st.stop()

shots      = gallery.get("shots", [])
total      = gallery.get("total_shots", 0)
reviewed   = gallery.get("reviewed_count", 0)
selections: dict = st.session_state.setdefault("selections", {})
img_cache: dict  = st.session_state.setdefault("img_cache", {})

if total == 0:
    filter_msg = f" (scene filter: {scene_input.strip()})" if scene_input.strip() else ""
    st.warning(f"No shots found for movie `{movie_id}`{filter_msg}. "
               "Check the Movie ID or try a different scene filter.")
    st.stop()

# Progress bar
pct = reviewed / total if total else 0
col_prog, col_filter = st.columns([4, 1])
with col_prog:
    st.progress(pct, text=f"{reviewed} / {total} shots reviewed  ({int(pct * 100)}%)")
with col_filter:
    hide_reviewed = st.checkbox("Hide reviewed shots")

st.divider()

# Group shots by scene for display
scenes: dict[int | None, list] = {}
for shot in shots:
    sn = shot.get("scene_number")
    scenes.setdefault(sn, []).append(shot)

for scene_num, scene_shots in sorted(scenes.items(), key=lambda x: (x[0] is None, x[0])):
    scene_label = f"Scene {scene_num}" if scene_num is not None else "Unassigned Scene"
    st.subheader(scene_label)

    for shot in scene_shots:
        shot_id  = shot["shot_id"]
        versions: dict = shot.get("versions") or {}
        already_selected = shot.get("selected") is not None

        if hide_reviewed and already_selected:
            continue

        all_images = [
            (v, idx, url)
            for v, urls in sorted(versions.items())
            for idx, url in enumerate(urls)
            if url
        ]

        seq   = shot.get("sequence_number")
        label = ("✅ " if already_selected else "") + f"Shot {seq or '?'}  —  {shot_id}"

        with st.expander(label, expanded=not already_selected):
            desc = shot.get("description", "")
            if desc:
                st.caption(desc)

            cur = selections.get(shot_id) or shot.get("selected")
            if cur:
                st.success(f"Selected: **{cur['version']}** / index {cur['index']}")

            if not all_images:
                st.warning("No generated images found for this shot.")
                continue

            COLS = 4
            for chunk_start in range(0, len(all_images), COLS):
                chunk = all_images[chunk_start : chunk_start + COLS]
                cols = st.columns(len(chunk))
                for col, (v, idx, url) in zip(cols, chunk):
                    with col:
                        if url not in img_cache:
                            img_cache[url] = fetch_image_bytes(url)
                            st.session_state["img_cache"] = img_cache

                        img_data = img_cache.get(url)
                        if img_data:
                            st.image(io.BytesIO(img_data), width="stretch",
                                     caption=f"{v} [{idx}]")
                        else:
                            st.error(f"Could not load\n{v}[{idx}]")

                        is_cur = cur and cur.get("version") == v and cur.get("index") == idx
                        btn_label = "✔ Selected" if is_cur else f"Select {v}[{idx}]"
                        if st.button(btn_label, key=f"{shot_id}_{v}_{idx}"):
                            selections[shot_id] = {"version": v, "index": idx, "url": url}
                            st.session_state["selections"] = selections
                            st.rerun()

st.divider()

col_save, col_clear = st.columns([2, 1])
with col_save:
    if st.button("Save All Selections", type="primary"):
        saved = failed = 0
        with st.spinner("Saving…"):
            for sid, sel in selections.items():
                ok = save_selection(api_url, admin_key, movie_id,
                                    sid, sel["version"], sel["index"], sel["url"])
                if ok:
                    saved += 1
                else:
                    failed += 1
        if saved:
            st.success(f"Saved {saved} selection(s).")
        if failed:
            st.error(f"{failed} selection(s) failed.")
        if saved:
            if master_job_id.strip():
                with st.spinner("Signalling pipeline to continue to Phase 3…"):
                    ok = continue_to_phase3(api_url, admin_key, master_job_id.strip())
                if ok:
                    st.success("Pipeline resumed — Phase 3 is now running.")
            data = fetch_gallery(api_url, admin_key, movie_id, _parse_int(scene_input))
            if data:
                st.session_state["gallery"] = data
            st.rerun()

with col_clear:
    if st.button("Clear & Reload"):
        for k in ("gallery", "selections", "img_cache"):
            st.session_state.pop(k, None)
        st.rerun()
