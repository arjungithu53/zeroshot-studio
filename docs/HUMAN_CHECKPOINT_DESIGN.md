# Human Image Review Checkpoint — Implementation Plan

## Problem

After Phase 2 generates multiple versions of each shot's first-frame image, the pipeline needs a human to pick the best one before Phase 3 starts video generation. Currently there is no mechanism for this — Phase 3 just picks the latest image automatically, which is not always the best one.

## Solution Overview

1. **Data model** — Add an `image.selected` field to each shot document so a human's choice is persisted in MongoDB.
2. **Two new API endpoints** — One to fetch all shots with their image galleries, one to save a selection.
3. **Streamlit review app** — A minimal single-file UI that calls those two endpoints, displays images side-by-side per shot, and lets a human click to select.
4. **Phase 3 priority update** — Before falling back to "latest image", Phase 3 checks for a human selection first.

---

## Part 1: Data Model Change

### Field to add to each `annotated_shots[]` entry in the `shots` collection

```json
"image": {
  "v0": { "generated_images_s3": ["url1", "url2"], ... },
  "v1": { "generated_images_s3": ["url3"],          ... },

  "selected": {
    "version": "v1",
    "index":   0,
    "url":     "https://s3.../url3.png",
    "selected_by": "human",
    "selected_at": "2026-05-25T17:30:00"
  }
}
```

- `version` — which `vN` key the human chose from
- `index` — which entry in that version's `generated_images_s3` array
- `url` — denormalised copy of the URL (fast access, no re-lookup needed)
- `selected_by` — `"human"` (reserved for future AI pre-selection)
- `selected_at` — ISO timestamp

No migration needed — the field simply won't exist on unreviewed shots, and Phase 3 falls back gracefully.

---

## Part 2: New `ShotsService` Method

**File:** `backend/services/production/app/services/shots_service.py`

Add one method after `update_shot_image_version` (~line 593):

```python
def set_shot_image_selection(
    self,
    show_id: str,
    episode_number: int,
    shot_id: str,
    version: str,          # e.g. "v1"
    index: int,            # index in generated_images_s3
    url: str,              # the chosen S3 URL
    selected_by: str = "human"
) -> bool:
```

**Implementation pattern** — follows the existing `update_shot_image_version` pattern:
- Use `$set` with `arrayFilters` on `annotated_shots.$[elem].image.selected`
- Same two-step null-init guard already used in `save_video_to_mongodb` (phase_3_agents/langgraph_workflow.py lines 258-279)
- Returns `True` if `matched_count > 0`

---

## Part 3: New API Endpoints

**File:** `backend/services/production/app/api/v1/endpoints/phase2.py`

Add two endpoints after the existing `GET /mongodb/shots/{show_id}/{episode_number}` (~line 1010).

---

### Endpoint A — Get image review gallery

```
GET /phase2/image-review/{show_id}/{episode_number}
```

**Response shape:**
```json
{
  "show_id": "...",
  "episode_number": 1,
  "total_shots": 24,
  "reviewed_count": 10,
  "shots": [
    {
      "shot_id": "S01E01_001",
      "scene_number": 1,
      "sequence_number": 1,
      "description": "...",
      "generation_strategy": "generate_new",
      "versions": {
        "v0": ["https://s3.../img1.png", "https://s3.../img2.png"],
        "v1": ["https://s3.../img3.png"]
      },
      "selected": {
        "version": "v1",
        "index": 0,
        "url": "https://s3.../img3.png",
        "selected_at": "2026-05-25T17:30:00"
      }
    }
  ]
}
```

**Logic:**
1. Call `shots_service.get_shots_from_atlas(show_id, episode_number)`
2. For each shot, extract `image` dict: collect all `vN` keys and their `generated_images_s3` arrays
3. Include `image.selected` if present, else `null`
4. Compute `reviewed_count` = shots where `selected` is not null

---

### Endpoint B — Save image selection

```
POST /phase2/image-review/{show_id}/{episode_number}/{shot_id}/select
```

**Request body:**
```json
{
  "version": "v1",
  "index": 0,
  "url": "https://s3.../img3.png"
}
```

**Response:**
```json
{ "success": true, "shot_id": "S01E01_001", "selected_url": "https://..." }
```

**Logic:**
- Validate that `version` exists and `index` is within bounds (optional guard)
- Call `shots_service.set_shot_image_selection(...)`
- Return success/failure

Both endpoints use the existing `validate_admin_from_header` dependency (same as all other phase2 endpoints).

---

## Part 4: Phase 3 Priority Update

**File:** `backend/services/production/app/services/phase_3_agents/video_generation/video_generation_api_agent.py`

In `_fetch_image_from_shots`, add a **Priority 0** check before the existing version-key logic (currently around line 393):

```python
# Priority 0: Human selection (most authoritative)
image_obj = shot.get("image", {})
if isinstance(image_obj, dict):
    selected = image_obj.get("selected", {})
    if isinstance(selected, dict) and selected.get("url"):
        url = selected["url"]
        logger.info(f"✅ Using human-selected image for shot {shot_id}: {url}")
        return url
```

Apply the same Priority 0 block in:
- `video_generation_api_agent.py` — `_fetch_image_from_shots` (annotated_shots path AND standalone doc path)
- `agent_video_generation.py` — `generate_video_prompt` `generate_new` branch and `get_seed_shot_s3_url`

**Priority order becomes:**
```
0. image.selected.url          ← human pick
1. image.{latestVN}.generated_images_s3[-1]   ← latest auto
2. root generated_images_s3[-1]               ← legacy format
3. image_s3_url                               ← direct field
```

---

## Part 5: Streamlit Review App

**File:** `tools/image_review.py` (new file, run standalone)

### Setup

```bash
pip install streamlit requests
streamlit run tools/image_review.py
```

### App layout

```
┌─────────────────────────────────────────────────────┐
│  🎬 Shot Image Review                               │
│  API URL: [____________]  Admin Key: [____________]  │
│  Show ID: [____________]  Episode:   [__]  [Load]   │
├─────────────────────────────────────────────────────┤
│  Progress: 10 / 24 shots reviewed  ████░░░░  42%   │
│  [ Show only unreviewed ]                           │
├─────────────────────────────────────────────────────┤
│  S01E01_001 — Scene 1, Shot 1                       │
│  "Character walks into the office..."               │
│                                                     │
│  v0                    v1                           │
│  [img]  [img]          [img]                        │
│  ○ Select  ○ Select    ○ Select     ← radio per img │
│                                                     │
│  ✅ Currently selected: v1 / index 0                │
├─────────────────────────────────────────────────────┤
│  S01E01_002 — Scene 1, Shot 2  ...                  │
└─────────────────────────────────────────────────────┘
         [ Save All Selections ]
```

### Key implementation details

- **Config sidebar**: API base URL, admin key, show_id, episode_number — stored in `st.session_state` so they persist across rerenders
- **Load button**: calls `GET /phase2/image-review/{show_id}/{episode_number}`, caches result in session_state
- **Image display**: `st.image(url, use_column_width=True)` inside `st.columns()` — one column per image
- **Selection**: `st.radio` or a button per image. Selection is tracked locally in `st.session_state["selections"][shot_id]`
- **Save All**: iterates `session_state["selections"]`, calls `POST .../select` for each changed shot
- **Already-reviewed indicator**: green checkmark next to shot header if `selected` is not null in the API response
- **Filter**: checkbox to hide already-reviewed shots (reduces noise when resuming a session)
- **No auth complexity**: admin key is just passed as an `X-Admin-Key` header on every request (same pattern as Swagger)

### File structure

```
tools/
  image_review.py     ← single self-contained Streamlit file (~200 lines)
  requirements.txt    ← streamlit, requests (minimal)
```

---

## Part 6: Workflow Integration

### When does the checkpoint happen?

```
Phase 2 completes
    ↓
Human opens Streamlit app
    ↓
Reviews and selects best image per shot  (saves via API as they go)
    ↓
Triggers Phase 3 via Swagger / existing POST /phase3/start
    ↓
Phase 3 reads image.selected.url — uses human pick
```

No new orchestration needed. The human checkpoint is purely advisory — Phase 3 will always fall back to latest auto-generated if no selection exists, so shots can be partially reviewed and Phase 3 still runs.

### Optional: Phase 3 guard

If you want Phase 3 to refuse to start unless all shots are reviewed, add a preflight check in `run_phase3_pipeline` or the Phase 3 endpoint:

```python
if require_review:
    unreviewed = [s for s in shots if not s.get("image", {}).get("selected")]
    if unreviewed:
        raise ValueError(f"{len(unreviewed)} shots have no human image selection")
```

This is a flag — off by default, can be enabled per-call.

---

## Implementation Order

| Step | File | Est. time |
|------|------|-----------|
| 1 | `shots_service.py` — add `set_shot_image_selection` | 30 min |
| 2 | `phase2.py` — add GET + POST endpoints | 45 min |
| 3 | `video_generation_api_agent.py` — Priority 0 block | 15 min |
| 4 | `agent_video_generation.py` — Priority 0 block | 15 min |
| 5 | `tools/image_review.py` — Streamlit app | 2–3 hrs |

**Total: ~4–5 hours**

---

## Files to Create / Modify

| Action | File |
|--------|------|
| Modify | `backend/services/production/app/services/shots_service.py` |
| Modify | `backend/services/production/app/api/v1/endpoints/phase2.py` |
| Modify | `backend/services/production/app/services/phase_3_agents/video_generation/video_generation_api_agent.py` |
| Modify | `backend/services/production/app/services/phase_3_agents/video_prompt_A/agent_video_generation.py` |
| Create | `tools/image_review.py` |
| Create | `tools/requirements.txt` |

---

## Verification

1. Run Phase 2 on an episode with multiple image versions per shot
2. Open Streamlit app → confirm all shots load with correct image counts
3. Select images for 2–3 shots → click Save → confirm API returns 200
4. Check MongoDB directly: `db.shots.findOne({"annotated_shots.shot_id": "S01E01_001"})` — verify `image.selected` is populated
5. Trigger Phase 3 for one of the reviewed shots → check logs for `✅ Using human-selected image` line
6. Trigger Phase 3 for an unreviewed shot → check logs show fallback to latest auto version
