# Production Pipeline — Phase 4 (Post-Production / Editing) Product Documentation

**Scope:** `backend/services/production/app/services/phase_4_agents/`
This document extends the existing production AI pipeline (Phases 1–3) with **Phase 4 — Post-Production**: taking the approved per-shot videos from Phase 3 and assembling them into a finished, voiced, scored, platform-ready Direct-Response (DR) ad. It covers every node/agent, system prompt, input/output schema, storage pattern, model, naming convention, **and a paste-ready Build Prompt for each agent**.

It is written to be drop-in consistent with the Phase 1–3 conventions: **LangGraph `StateGraph`**, **Pydantic v2**, **AWS S3 presigned URLs (7-day expiry)**, **MongoDB Atlas**, **Celery (fork model — fresh DB connections per node)**.

---

## Table of Contents

1. System Overview
   - Purpose & where Phase 4 fits
   - High-level flow
   - What was added vs. your described plan (design decisions)
   - Tech stack additions
   - New MongoDB collection (`final_assemblies`)
   - Environment variables (additions)
   - `Phase4State` schema
   - Storage patterns (Phase 4)
   - Phase 3 video storage & clip referencing (UPDATED to match real S3 + Mongo)

**Agent 0 — Human Review Checkpoint (Phase 3 → Phase 4 gate)** — Streamlit `video_review.py` + `/api/v1/phase4` endpoints

2. Initialize Node
3. Agent 1 — EDL Generator (DR Editor)
4. Agent 2 — Assembly / Cutter (FFmpeg)
5. Agent 3 — Final-Cut Review
6. Agent 4 — Timestamp Reviser *(optional, loop)*
7. Agent 5 — Voiceover (VO) Agent *(Director + TTS, two-step)*
8. Agent 6 — A/V Merge (VO Preview Cut)
9. Agent 7 — Music / SFX Agent *(Director + Lyria, two-step)*
10. Agent 8 — Final Mix & Master *(NEW)*
11. Agent 9 — Delivery & Captions *(NEW, optional)*
12. LangGraph Workflow Wiring (`langgraph_workflow.py`)
13. Cross-Cutting Infrastructure (S3 keys, MongoDB writes, loop caps)
14. Naming Conventions & Nomenclature (Phase 4 additions)

---

## 1. System Overview

### Purpose & Where Phase 4 Fits

```
Phase 1  ─► Asset Images + Angle Variations (S3)
Phase 2  ─► Per-Shot Composite Images (S3)
Phase 3  ─► Per-Shot Videos (S3)              ◄── Phase 4 INPUT
Phase 4  ─► One Finished Ad: cut + VO + music + master + platform exports (S3)
```

Phase 3 produces **one or several versioned video clips per shot**, stored in the `zeroshot-v1` bucket (region `eu-north-1`) at `phase3/{project_folder}/generated_videos/scene_{scene}_shot_{shot}_v{n}.mp4` — e.g. `phase3/mama_earth_1_6a14e1ef57ecca6e03378d9b/generated_videos/scene_1_shot_1_v2.mp4`. Each version's reference is stored in MongoDB as a **presigned URL** at `shots.video.v{N}.generated_videos_s3[]`; the exact, durable S3 key is **parsed out of that URL** (never reconstructed from a pattern). **Note:** filename versions start at **`_v1`** (not `v0`), and the MongoDB attempt key (`video.v0`, `video.v1`, …) is the *generation attempt* — it may not equal the filename's `_v{n}`, so we always key off the literal filename in the stored URL.

**Phase 4 is invoked once per episode/ad** (not per shot). Before it runs, a human passes through **Agent 0 — the Human Review Checkpoint** (Streamlit `video_review.py`), choosing which clip version(s) to use per shot; Phase 4 then consumes those approved clips plus the **script** and **shot list** to build the final video.

### High-Level Flow

```
[Agent 0 — Human Review Checkpoint]  (Streamlit video_review.py; human picks 0/1/many versions per shot → saved to Mongo → POST /master/continue-to-phase4 starts the graph below)
 │
 ▼
initialize_node  (read checkpoint selections; resolve clips → candidate-group manifest; load script + shotlist)
 └─► agent_1_edl_node           (Gemini 3.1 Pro, video understanding → Edit Decision List)
      └─► agent_2_assembly_node (FFmpeg: trim each clip by exact name+version, normalize, concat → rough cut)
           └─► agent_3_review_node (Gemini 3.1 Pro: review the rough cut)
                ├─► agent_4_revise_node ─► agent_2_assembly_node   (if "edit", loop, MAX 2)
                └─► agent_5_vo_node      (if "approved" or force-pass)
                     │   ├─ Director call:  Gemini 3.1 Pro (video) → optimized script + TTS prompt
                     │   └─ TTS call:       Gemini 3.1 Flash TTS  → single-take VO audio (S3)
                     └─► agent_6_av_merge_node    (FFmpeg: video + VO → VO Preview Cut, S3)
                          └─► agent_7_music_node  (two-step)
                          │     ├─ Director call: Gemini 3.1 Pro (video) → filled Lyria prompt
                          │     └─ Lyria call:    Lyria 3 → instrumental score (S3)
                          └─► agent_8_final_mix_node  (FFmpeg: duck music under VO, loudness master, mux → FINAL, S3)
                               └─► agent_9_delivery_node  (captions + platform exports + metadata)  [optional]
                                    └─► END
```

### What Was Added vs. Your Described Plan (Design Decisions)

You said new agents are welcome if they improve the workflow. These are the deltas — each is optional to keep, but recommended:

1. **Agent 8 — Final Mix & Master (NEW, strongly recommended).** Your plan ends with Agent 7 *generating* a music track but never integrates it. A finished ad needs the music **ducked under the VO** (sidechain compression) and the whole thing **loudness-normalized** for social (~−14 LUFS). Agent 8 does the mix + master + final mux. Without it you have an orphan music file.
2. **Agent 5 and Agent 7 are each two-step ("Director → Generator").** The system prompts you supplied for VO and Music are **analysis/authoring prompts** (they require *seeing the final video* and *writing* a script / filling a template). A TTS model (Flash TTS) and a music model (Lyria) cannot do video understanding — they only take a finished prompt. So each of these agents runs a **Director call** on **Gemini 3.1 Pro (multimodal/video)** to produce the generation prompt, then a **Generator call** on the actual media model. This is the only correct way to wire those prompts; it is documented explicitly below.
3. **Agent 9 — Delivery & Captions (NEW, optional).** DR ads are watched **muted**; burned-in captions materially lift retention. Captions are generated **deterministically** from Agent 5's optimized script + slice-guide timestamps (no extra model), plus per-platform exports and final metadata.
4. **Strict version disambiguation enforced end-to-end.** Per your repeated note: a shot's S3 folder holds several versions (`scene_2_shot_1_v1.mp4`, `_v2.mp4`, `_v3.mp4`). Every clip reference in the EDL, the cutter, and all downstream agents carries the **exact `s3_key` + `version` + `filename`** — never a fuzzy description like "the gym shot." The `s3_key` is **parsed from the stored presigned URL** (`shots.video.v{N}.generated_videos_s3[]`), never reconstructed, and the cutter fetches **by exact key**. See the `ClipRef` schema and the rule boxes in Agents 1–2.
5. **Presigned URLs are always regenerated from `s3_key`.** Phase 3 URLs expire after 7 days; by the time Phase 4 runs they may be dead. `initialize_node` and every node regenerate fresh presigned URLs from the stored `s3_key`.
6. **System prompts written where you didn't supply one:** Agent 3 (Review), Agent 4 (Timestamp Reviser), and the Director sub-step of Agent 7 (Music Director) get full, detailed system prompts below. The three you supplied (Agent 1 DR Editor, Agent 5 Audio Director, Agent 7 Lyria template) are reproduced **verbatim** (the Audio Director list numbering is normalized 1–6; wording unchanged).

7. **Agent 0 — Human Review Checkpoint (NEW, per your request).** A Streamlit gate (`tools/video_review.py`, port 8004) between Phase 3 and Phase 4, mirroring your Phase 2→3 image checkpoint. The reviewer picks **0, 1, or many** versions per shot: 1 → used directly; 2+ → Agent 1 picks one; 0 (or skip) → fallback (a), Agent 1 picks among all versions. This turns the manifest into per-shot **candidate groups** (`ShotCandidates`) and upgrades Agent 1 to choose within them. Backend endpoints under `/api/v1/phase4` are specified in the Agent 0 section.

### Tech Stack Additions

| Component | Technology |
|---|---|
| Workflow orchestration | LangGraph (`StateGraph`, `CompiledStateGraph`) — same as P1–P3 |
| LLM for EDL / review / revision / directing | `gemini-3.1-pro-preview` (video understanding) via `google.genai` |
| Voiceover (TTS) | `gemini-3.1-flash-tts-preview` (Gemini 3.1 Flash TTS) via `google.genai` |
| Music / score | `lyria-3` (Lyria 3) via `google.genai` |
| Video cut / concat / mux / mix | **FFmpeg** (invoked via `subprocess`), `ffprobe` for duration |
| Audio loudness / ducking | FFmpeg filters: `sidechaincompress`, `loudnorm`, `amix`, `aformat` |
| Video/Audio storage | AWS S3 (presigned URLs, 7-day expiry) |
| Metadata storage | MongoDB Atlas (`final_assemblies` + `production_pipelines`) |
| Job queue | Celery (fork model — fresh DB connections per node) |
| Data validation | Pydantic v2 |

> **FFmpeg vs. alternatives.** FFmpeg via `subprocess` is the recommended engine: fastest, frame-accurate when re-encoding, and the only tool with first-class loudness/sidechain filters. `MoviePy` wraps FFmpeg but is slower, leakier on long jobs, and weaker on audio mastering; `PyAV` gives fine control but is far more code. **Use FFmpeg.** Pin a known build in the Celery worker image and verify on boot (`ffmpeg -version`).

### New MongoDB Collection

| Collection | Purpose |
|---|---|
| `final_assemblies` | Phase 4 ad documents, keyed by `{show_id, episode_number, episode_id}`. Holds the EDL (versioned), rough-cut versions, review results, VO/music/preview/final versions, and deliverables. Mirrors the `agent_outputs.agent{N}` pattern used elsewhere. |
| `production_pipelines` | Reused as the lightweight job tracker (`agent{N}_status`, `current_agent`, `pipeline_status`) under the Phase 4 `job_id`. |
| `shots` | Source of Phase 3 videos. Each shot (one document per shot; or an entry under `annotated_shots[]` if your build nests them) holds `video.v{N}.generated_videos_s3[]` — presigned URLs whose path encodes the exact key + `scene_{N}_shot_{M}_v{V}` filename — plus `video.v{N}.approval_status` / `approved_at`. **Agent 0 writes the human's chosen version(s) here** as `video_review_selection`, read by `initialize_node`. |
| `movies` / `production_projects` (read-only here) | Source of `title`, `visual_style`, and episode context. |

`final_assemblies` document shape (high level):

```
{
  "_id": ObjectId,
  "show_id": "abc123",
  "episode_number": 1,
  "episode_id": "S01E01",
  "movie_id": "...",
  "title": "GLOW SERUM LAUNCH",
  "clip_manifest": [ ClipRef, ... ],
  "agent_outputs": {
    "agent1": { "status": "completed", "executed_at": ISODate, "output": { "edl": {...}, "edl_version": 0 } },
    "agent2": { "status": "completed", "executed_at": ISODate, "output": { "rough_cut": {...} } },
    ...
  },
  "edl_versions":         [ { "version": 0, "edl": {...}, "created_at": ISODate }, ... ],
  "rough_cuts":           [ { "version": 0, "s3_key": "...", "s3_url": "...", "duration": 18.4, "created_at": ISODate }, ... ],
  "reviews":              [ { "for_version": 0, "decision": "edit", "score": 72, ... }, ... ],
  "vo":                   [ { "version": 0, "s3_key": "...", "plan": {...}, "duration": 17.9 }, ... ],
  "vo_preview":           [ { "version": 0, "s3_key": "...", "s3_url": "..." }, ... ],
  "music":                [ { "version": 0, "s3_key": "...", "prompt": "...", "duration": 18.0 }, ... ],
  "final_masters":        [ { "version": 0, "s3_key": "...", "s3_url": "...", "lufs": -14.0 }, ... ],
  "deliverables":         { "captions_srt": "s3_key", "exports": { "reels": "s3_url", "tiktok": "s3_url" } },
  "pipeline_status": "completed",
  "updated_at": ISODate
}
```

### Environment Variables (Additions)

```
GEMINI_API_KEY / GOOGLE_API_KEY     # Gemini Pro, Flash TTS, Lyria (reused)
production_AWS_ACCESS_KEY_ID         # S3 (reused)
production_AWS_SECRET_ACCESS_KEY     # S3 (reused)
production_S3_BUCKET_NAME            # S3 (reused) — observed: "zeroshot-v1"
production_AWS_REGION                # S3 (reused) — observed: "eu-north-1"
MONGODB_ATLAS_URI                   # Atlas (reused)

# Phase-4 specific
PHASE4_FFMPEG_PATH                  # default: "ffmpeg"
PHASE4_FFPROBE_PATH                 # default: "ffprobe"
PHASE4_TARGET_ASPECT_RATIO          # default: "9:16"  (DR social)
PHASE4_TARGET_RESOLUTION            # default: "1080x1920"
PHASE4_TARGET_FPS                   # default: "30"
PHASE4_TARGET_LUFS                  # default: "-14"   (integrated loudness)
PHASE4_TARGET_TP                    # default: "-1.5"  (true peak dBTP)
PHASE4_FIT_MODE                     # default: "crop"  ("crop" | "pad" | "blur_pad")
PHASE4_TMP_DIR                      # default: "/tmp/phase4"

# Agent 0 — human review checkpoint (Streamlit tool + endpoints)
PHASE4_API_BASE                     # default: "http://localhost:8000/api/v1/phase4"
PHASE4_REVIEW_PORT                  # default: "8004"  (Streamlit video_review.py)
```

### `Phase4State` Schema

`File: phase_4_agents/workflow_state.py`

```python
from typing import TypedDict, Optional, List, Dict, Any

class Phase4State(TypedDict):
    # ---- Identifiers ----
    show_id: str
    episode_number: int
    episode_id: str
    movie_id: Optional[str]
    project_id: Optional[str]
    title: str                      # project/movie/episode title (used in S3 filenames)
    job_id: Optional[str]

    # ---- Inputs (loaded by initialize_node) ----
    script_content: str
    shot_list: Dict                 # raw or annotated shot list (guideline, not law)
    clip_manifest: List[Dict]       # List[ShotCandidates] — per-shot candidate group(s); 1 clip = pre-resolved, >1 = Agent 1 chooses one
    aspect_ratio: str               # final output AR, default from env
    target_platforms: List[str]     # e.g. ["reels","tiktok","shorts"]

    # ---- Agent 1: EDL ----
    edl: Dict                       # EditDecisionList (overall_strategy, clips[], loop_check, est_length)
    edl_version: int                # 0,1,2...

    # ---- Agent 2: Assembly ----
    rough_cut_s3_url: str
    rough_cut_s3_key: str
    rough_cut_version: int
    assembled_duration: float
    trimmed_clip_keys: List[Dict]   # optional per-segment refs persisted for debugging

    # ---- Agent 3: Review ----
    review_result: Dict             # FinalCutReview
    review_decision: str            # "approved" | "edit"
    edit_loop_count: int            # max 2

    # ---- Agent 4: Timestamp revision ----
    revised_edl: Dict               # updated EditDecisionList

    # ---- Agent 5: VO ----
    vo_plan: Dict                   # Audio Director structured output (persona, breakdown, word-math, script, slice guide)
    vo_generation_prompt: Dict      # {"text_to_speak": str, "style_instructions": str}
    vo_s3_url: str
    vo_s3_key: str
    vo_version: int
    vo_duration: float

    # ---- Agent 6: A/V merge (VO preview) ----
    vo_preview_s3_url: str
    vo_preview_s3_key: str
    vo_preview_version: int

    # ---- Agent 7: Music / SFX ----
    music_prompt: str               # filled Lyria prompt
    music_plan: Dict                # Music Director rationale + timing map
    music_s3_url: str
    music_s3_key: str
    music_version: int
    music_duration: float

    # ---- Agent 8: Final mix & master ----
    final_master_s3_url: str
    final_master_s3_key: str
    final_master_version: int
    loudness_lufs: float

    # ---- Agent 9: Delivery (optional) ----
    captions_s3_key: Optional[str]
    platform_exports: Optional[Dict]   # {platform: s3_url}
    final_metadata: Optional[Dict]

    # ---- Control ----
    current_agent: str
    pipeline_status: str
    errors: List[Dict]
```

`ClipRef` (the canonical clip reference — the version-disambiguation backbone). The `s3_key` is **parsed from the stored presigned URL** in `shots.video.v{N}.generated_videos_s3[]`, and `version` is taken from the **literal filename** (`scene_{N}_shot_{M}_v{V}.mp4` → `"v2"`), not the Mongo attempt key:

```python
class ClipRef(TypedDict):
    shot_id: str            # "scene_1_shot_1"  (canonical: scene_{N}_shot_{M})
    scene_number: int       # 1   (parsed from filename / shot doc)
    shot_number: int        # 1   (the shot_{M} index within the scene)
    version: str            # "v1" | "v2" | "v3" ...  (FILENAME version, starts at v1)
    attempt_key: str        # "v0" | "v1" ...  (the MongoDB video.v{N} generation-attempt key; traceability only)
    s3_key: str             # EXACT, parsed from the stored URL: "phase3/mama_earth_1_6a14.../generated_videos/scene_1_shot_1_v2.mp4"
    s3_url: str             # FRESH presigned URL re-minted from s3_key (the stored one may be expired)
    filename: str           # "scene_1_shot_1_v2.mp4"
    description: str        # from shot doc (LLM context only)
    duration: float         # seconds (ffprobe-measured)
    has_audio: bool         # whether the Veo clip carries usable audio
    approval_status: str    # the shot version's status ("approved" | "pending" | ...)
```

`ShotCandidates` (what the manifest is actually a list of — one entry per shot, carrying the human's checkpoint decision):

```python
class ShotCandidates(TypedDict):
    shot_id: str                 # "scene_1_shot_1"
    scene_number: int
    shot_number: int
    selection_mode: str          # "single"     → use candidates[0] as-is (human picked exactly one)
                                 # "choose_one" → Agent 1 MUST pick exactly one of candidates
                                 #   (human picked 2+, OR human picked none → fallback: all versions)
    selection_source: str        # "human" | "fallback_all_versions"
    candidates: List[ClipRef]    # 1 if "single"; >=1 if "choose_one"
```

**Selection → manifest mapping (set by `initialize_node` from Agent 0's saved selection):**

| Human picked for a shot | `selection_mode` | `candidates` |
|---|---|---|
| exactly 1 version | `single` | that one ClipRef |
| 2+ versions | `choose_one` | the chosen ClipRefs (Agent 1 picks one) |
| 0 versions (or skipped the whole checkpoint) | `choose_one` | **all** of that shot's versions (fallback (a) — LLM chooses) |

### Storage Patterns (Phase 4)

Every node:
1. Regenerates presigned URLs from `s3_key` (never reuses stale URLs).
2. Writes its artifact to S3 under the Phase 4 key patterns (see §13).
3. Calls `final_assemblies_service.update_agent_output(show_id, episode_number, agent_number=N, status="completed", output={...})` mirroring the Phase 1–3 `agent_outputs.agent{N}` pattern.
4. Pushes job status via `PipelineService.update_job_status(job_id, agent_number=N, status=...)` to `production_pipelines`.

---

## Agent 0 — Human Review Checkpoint (Phase 3 → Phase 4 Gate)

> **Not a LangGraph node.** Agent 0 is the human gate that sits **between Phase 3 and Phase 4**, exactly like the existing Phase 2→3 image checkpoint (`tools/image_review.py`). It is a Streamlit app plus three backend endpoints. A reviewer opens it after Phase 3 finishes, watches every shot's video versions, **selects which version(s) to use per shot**, saves, and (optionally) triggers Phase 4. The Phase 4 graph itself still starts at `initialize_node`, which reads what Agent 0 saved.

| Field | Value |
|---|---|
| Streamlit tool | `tools/video_review.py` (run: `streamlit run tools/video_review.py --server.port 8004`) |
| Backend base | `/api/v1/phase4` |
| Endpoints | `GET /video-review/{movie_id}`, `POST /video-review/{movie_id}/{shot_id}/select`, `POST /master/continue-to-phase4/{master_job_id}` |
| Writes to | `shots.<shot>.video_review_selection` (read by `initialize_node`) |
| Model | None (human-in-the-loop) |

### Selection semantics (the rule)

Per shot, the reviewer may select **0, 1, or many** versions:

| Reviewer picks | What Phase 4 does |
|---|---|
| **exactly 1** version | use it directly (no LLM choice) |
| **2 or more** versions | pass those candidates to **Agent 1 (EDL)**, which picks exactly one |
| **0** versions (or skips the whole checkpoint) | **fallback (a):** pass **all** of that shot's versions to Agent 1, which picks one |

So selection is **fully optional** — a reviewer who saves nothing still gets a valid Phase 4 run (every shot defaults to "LLM chooses among all versions"). This is why Agent 1 is upgraded to handle **candidate groups** (see §3) and why `initialize_node` materializes `ShotCandidates` rather than a flat one-clip-per-shot list.

### How it mirrors the Phase 2 image checkpoint

`video_review.py` reuses the exact structure of `tools/image_review.py`, swapping images→videos and single-select→multi-select:

- **Sidebar config:** API Base URL (default `http://localhost:8000/api/v1/phase4`), Admin Key, Movie ID, Master Job ID (optional — used to auto-trigger Phase 4 after saving), Scene-number filter (blank = all scenes).
- **Load Gallery:** `GET /video-review/{movie_id}?scene_number=N` → shots grouped by scene, each shot with its versioned videos (`v0`/`v1`/… attempts) and their **S3 presigned URLs**.
- **Display:** shots grouped by scene in expanders; each shot's versions in a grid; videos are **server-side proxied** through Python (`requests.get(s3_url)` → bytes → `st.video`) so private presigned URLs render without CORS issues; bytes cached in `st.session_state["vid_cache"]`.
- **Select:** per version a **checkbox** (multi-select, unlike images' single Select button); selections held in `st.session_state["selections"][shot_id]` as a list.
- **Save All Selections:** for each shot, `POST /video-review/{movie_id}/{shot_id}/select` with the chosen list, persisting to MongoDB.
- **Auto-continue to Phase 4:** if Master Job ID was filled in, after saving, `POST /master/continue-to-phase4/{master_job_id}` resumes the LangGraph pipeline (Phase 4).
- **Niceties:** "Hide reviewed shots" checkbox, progress bar (`{reviewed}/{total}`), no auto-trigger (Phase 4 fires only on explicit save **and** a Master Job ID present).

### Endpoint contracts

```
GET /api/v1/phase4/video-review/{movie_id}?scene_number={N|blank}
→ 200 {
    "movie_id": "...",
    "scenes": [
      { "scene_number": 1,
        "shots": [
          { "shot_id": "scene_1_shot_1",
            "scene_number": 1, "shot_number": 1,
            "versions": [
              { "version": "v1",            # FILENAME version (from the key)
                "attempt_key": "v0",        # MongoDB video.v{N} key
                "s3_key": "phase3/.../generated_videos/scene_1_shot_1_v1.mp4",
                "s3_url": "<fresh presigned>",
                "approval_status": "pending",
                "prompt": "<updated_prompt>" },
              ...
            ],
            "current_selection": ["v2"]     # echo of any saved video_review_selection
          }, ...
        ] }, ...
    ]
  }

POST /api/v1/phase4/video-review/{movie_id}/{shot_id}/select
body { "selected": [ { "version": "v2",
                       "attempt_key": "v1",
                       "s3_key": "phase3/.../scene_1_shot_1_v2.mp4",
                       "s3_url": "..." }, ... ] }   # 0, 1, or many
→ persists shots.<shot>.video_review_selection = {
      "selected": [...], "mode": "single"|"multi"|"none",
      "selected_by": "<admin>", "selected_at": ISODate() }
→ 200 { "ok": true, "shot_id": "...", "count": <n> }

POST /api/v1/phase4/master/continue-to-phase4/{master_job_id}
→ kicks off the Phase 4 LangGraph (run_phase4_pipeline) for the movie/episode
→ 202 { "ok": true, "job_id": "...", "phase": 4 }
```

> **Key parsing (shared rule).** The backend stores each video as a **presigned URL** in `shots.video.v{N}.generated_videos_s3[]`. To get the durable `s3_key`, strip the query string: everything after `https://s3.{region}.amazonaws.com/{bucket}/` and before `?`. Both the `GET` endpoint and `initialize_node` use this exact parse; presigned URLs are always **re-minted** from the key before use (the stored ones expire in 24h — `X-Amz-Expires=86400`).

The Streamlit `tools/video_review.py` is delivered as a standalone runnable file (it mirrors `tools/image_review.py`). Its build prompt:

````text
You are building **`tools/video_review.py`** — the Phase 3 → Phase 4 human review checkpoint, a Streamlit app. It mirrors the existing `tools/image_review.py` (the Phase 2→3 image checkpoint) but for VIDEOS and with MULTI-SELECT. Output one complete Python file.

PURPOSE
After Phase 3 generates per-shot videos, a human opens this app, watches each shot's video versions, picks which version(s) to use per shot (0, 1, or many), saves to MongoDB via the backend, and optionally triggers Phase 4.

RUN: `streamlit run tools/video_review.py --server.port 8004`.

SIDEBAR CONFIG (reuse image_review's variables/logic):
- API Base URL — default "http://localhost:8000/api/v1/phase4"
- Admin Key (sent as a header, e.g. X-Admin-Key)
- Movie ID
- Master Job ID (optional — if present, enables auto-continue to Phase 4 after save)
- Scene number filter (blank = all scenes)

FLOW
1. "Load Gallery" button → GET {API_BASE}/video-review/{movie_id} (+ ?scene_number=N if filter set), header X-Admin-Key. Response shape: { scenes: [ { scene_number, shots: [ { shot_id, scene_number, shot_number, versions: [ { version, attempt_key, s3_key, s3_url, approval_status, prompt } ], current_selection: [..] } ] } ] }.
2. Display shots grouped by scene in `st.expander` sections (one per scene). For each shot, show its versions in a column grid (e.g. 3 per row). For each version:
   - Render the video by SERVER-SIDE PROXYING the presigned URL: `requests.get(s3_url).content` → pass bytes to `st.video(...)`. Cache bytes in `st.session_state["vid_cache"][s3_key]` to avoid re-downloading on reruns. (Fall back to `st.video(s3_url)` if the proxy fetch fails.)
   - Show the version label (filename version + attempt_key) and approval_status.
   - A `st.checkbox` for "use this version" (MULTI-SELECT). Maintain `st.session_state["selections"][shot_id]` as a list of selected version dicts; checking/unchecking updates it.
3. "Hide reviewed shots" checkbox — hides shots that already have a non-empty selection. Progress bar showing reviewed/total shots.
4. "Save All Selections" button → for EACH shot with state, POST {API_BASE}/video-review/{movie_id}/{shot_id}/select with body { "selected": [ {version, attempt_key, s3_key, s3_url}, ... ] } (send even empty lists if the user explicitly cleared a shot). Header X-Admin-Key. Show per-shot success/failure.
5. If Master Job ID is set, after a successful save, call POST {API_BASE}/master/continue-to-phase4/{master_job_id} and surface the response. Do NOT auto-trigger without an explicit save + a Master Job ID (mirror image_review's "no automatic trigger" rule).

REQUIREMENTS: requests for HTTP, clean session_state init, defensive error handling with st.error on non-2xx, a "Reset selections" button, spinners on slow downloads, and code/variable names consistent with image_review.py so the two tools feel identical. Deliver the complete file.
````

**Backend endpoints build prompt:**

````text
You are adding the **Phase 4 human-review checkpoint endpoints** to the existing FastAPI backend (the same app that serves `/api/v1/phase2/image-review/...`). Mirror those image endpoints. Output the new router module.

ENDPOINTS (prefix `/api/v1/phase4`)
1. GET `/video-review/{movie_id}` (optional query `scene_number: int`)
   - Load the movie's shots from the `shots` collection (filter by scene_number if given). One document per shot (or iterate `annotated_shots[]` if nested).
   - For each shot, for each `video.v{N}` attempt, for each url in `video.v{N}.generated_videos_s3[]`:
     * parse the durable s3_key from the stored presigned URL (strip query string),
     * parse filename version from `scene_{N}_shot_{M}_v{V}.mp4`,
     * RE-MINT a fresh presigned GET url from the key (boto3, ExpiresIn=86400*7, bucket from production_S3_BUCKET_NAME, region production_AWS_REGION),
     * include attempt_key="v{N}", approval_status, and updated_prompt.
   - Group shots by scene. Echo any existing `video_review_selection` as `current_selection`. Return the shape documented above.
2. POST `/video-review/{movie_id}/{shot_id}/select`
   - body: { "selected": [ {version, attempt_key, s3_key, s3_url} ... ] } (0..n).
   - Persist `shots.<shot>.video_review_selection = {selected, mode: "none"|"single"|"multi", selected_by, selected_at}` (mode from len(selected)).
   - Return { ok, shot_id, count }.
3. POST `/master/continue-to-phase4/{master_job_id}`
   - Resolve the job/movie/episode, then enqueue/kick off `run_phase4_pipeline(...)` (Celery task or direct), mirroring how `continue-to-phase3` resumes the pipeline.
   - Return 202 { ok, job_id, phase: 4 }.

REQUIREMENTS: admin-key auth dependency (same as image-review), fresh Mongo/S3 clients per request or your existing DI, Pydantic response models, the SHARED key-parsing helper (strip query string; never reconstruct), structured errors. Deliver the complete router module.
````

---

## 2. Initialize Node

| Field | Value |
|---|---|
| Function | `initialize_node` |
| File | `phase_4_agents/langgraph_workflow.py` |
| Model | None (data gathering) |
| Entry point | Yes |

**Purpose.** Read **Agent 0's human selection** and resolve each shot into a `ShotCandidates` entry (one or more `ClipRef`s), load the script + shot list, set the title and output targets. This is where the checkpoint decision becomes the candidate-group `clip_manifest`.

**Resolution logic (per shot in the episode):**
1. Read the `shots` for `{show_id, episode_number}` (one document per shot, or `annotated_shots[]` if your build nests them).
2. Read the shot's `video_review_selection` (written by Agent 0). Determine the candidate set:
   - **1 version selected** → `selection_mode="single"`, `selection_source="human"`, candidates = that one version.
   - **2+ versions selected** → `selection_mode="choose_one"`, `selection_source="human"`, candidates = those versions (Agent 1 will pick one).
   - **0 selected / no selection present** (shot skipped, or whole checkpoint skipped) → **fallback (a)**: `selection_mode="choose_one"`, `selection_source="fallback_all_versions"`, candidates = **all** of the shot's `video.v{N}` versions.
3. For each candidate version: take `video.v{N}.generated_videos_s3[0]`, **strip the query string to get the exact `s3_key`** (everything after the bucket host, before `?`). Parse `scene_{N}`, `shot_{M}`, and the filename `_v{V}` from the key. Keep `attempt_key = "v{N}"` (the Mongo key) and `approval_status` for traceability.
4. **Re-mint** a fresh presigned GET URL from that exact `s3_key` (the stored URL is likely expired — `X-Amz-Expires=86400` = 24h).
5. Probe duration with `ffprobe`; detect audio stream presence (`has_audio`).
6. Build a `ClipRef` per candidate; assemble the `ShotCandidates`. **Order the manifest by `(scene_number, shot_number)`.**
7. Load `script_content` and `shot_list` (from the episode doc / passed args). Load `title` from `movies`/episode doc; compute `TITLE_SAFE` (see §14).
8. Set `aspect_ratio`, `target_platforms` from args/env defaults. Initialize all loop counters and Phase-4 output versions to 0.

**Storage:** writes the candidate-group `clip_manifest` to `final_assemblies` (creates the doc if absent); job status `running`. Always routes to `agent_1_edl_node`.

````text
You are building the **initialize_node** for Phase 4 (Post-Production) of an existing LangGraph video-production pipeline. Output a single complete Python module.

CONTEXT
- This is the entry node of `phase_4_agents/langgraph_workflow.py`. Phase 4 assembles approved per-shot videos (produced by Phase 3) into one finished ad. It runs once per episode.
- Stack: LangGraph StateGraph, Pydantic v2, boto3 (S3), pymongo (MongoDB Atlas), subprocess (ffprobe). Celery fork model: open fresh Mongo/S3 clients inside the node, do not rely on module-global connections.
- The state type is `Phase4State` and the clip reference type is `ClipRef` (both defined in `phase_4_agents/workflow_state.py`). Assume they are importable; I will paste their definitions below.

STATE TYPES (import these)
[Paste the `Phase4State`, `ClipRef`, and `ShotCandidates` TypedDicts from this document here.]

WHAT TO BUILD
A function `initialize_node(state: Phase4State) -> Phase4State` plus small private helpers.

INPUTS expected on entry: `show_id`, `episode_number`, `episode_id`, optional `movie_id`/`project_id`, optional `title`, `job_id`, and optionally `script_content`/`shot_list`/`aspect_ratio`/`target_platforms` (fill defaults from env if absent).

STEPS
1. Open fresh `MongoClient(os.environ["MONGODB_ATLAS_URI"])` and a fresh boto3 S3 client using `production_AWS_*` env vars (region default "eu-north-1", bucket from production_S3_BUCKET_NAME, observed "zeroshot-v1").
2. Read the `shots` for `{show_id, episode_number}` (one document per shot, or iterate `annotated_shots[]` if your build nests them).
3. For each shot, read its `video_review_selection` (written by Agent 0) and build the candidate set:
   - `len(selected) == 1` → selection_mode="single", selection_source="human", candidates = that one version.
   - `len(selected) >= 2` → selection_mode="choose_one", selection_source="human", candidates = those versions.
   - `selected` empty OR field missing → FALLBACK (a): selection_mode="choose_one", selection_source="fallback_all_versions", candidates = ALL of the shot's `video.v{N}` versions.
   (A "version" = one entry of `video.v{N}.generated_videos_s3[]`.)
4. For EACH candidate version: take its `generated_videos_s3[0]` presigned URL and PARSE THE EXACT S3 KEY from it — i.e. the path after the bucket host and before "?": `phase3/{project_folder}/generated_videos/scene_{N}_shot_{M}_v{V}.mp4`. Do NOT reconstruct the key from a pattern; the stored URL is the source of truth. Keep `attempt_key="v{N}"` (the Mongo key) and the version's `approval_status`.
5. Re-mint a FRESH presigned GET url from that exact s3_key with `ExpiresIn=86400*7` (the stored URL uses `X-Amz-Expires=86400` = 24h and is likely dead by Phase 4).
6. Download each candidate to `PHASE4_TMP_DIR` and run `ffprobe` for `duration` (float seconds) and audio-stream presence (`has_audio`). Use `ffprobe -v error -show_entries stream=codec_type,duration -of json`.
7. Parse `scene_number`/`shot_number`/filename-`version` from the filename via regex `scene_(\d+)_shot_(\d+)_v(\d+)\.mp4` (fall back to stored shot fields if needed). Build a `ClipRef` per candidate, set `shot_id="scene_{N}_shot_{M}"`, and assemble a `ShotCandidates` per shot. Sort the manifest by `(scene_number, shot_number)`. Set `state["clip_manifest"]`.
8. Load `script_content` and `shot_list` if not already on state. Load `title` from `movies`/episode doc; compute `TITLE_SAFE = title.upper().replace(" ", "_").replace("/", "_")`.
9. Set defaults from env: `aspect_ratio` (PHASE4_TARGET_ASPECT_RATIO), `target_platforms` (comma list, default ["reels","tiktok","shorts"]). Initialize: `edl_version=0`, `rough_cut_version=0`, `edit_loop_count=0`, `vo_version=0`, `music_version=0`, `vo_preview_version=0`, `final_master_version=0`, `errors=[]`, `pipeline_status="running"`, `current_agent="initialize"`.
10. Upsert a `final_assemblies` document keyed by `{show_id, episode_number, episode_id}` containing `clip_manifest`, `title`, identifiers, `pipeline_status:"running"`, `updated_at`. Create it if missing.
11. Push job status: set `production_pipelines` for this `job_id` → `current_agent:"initialize"`, `pipeline_status:"running"`.

OUTPUT: return the mutated `state`. Raise a clear exception (and record in `state["errors"]`) if zero shots/candidates were resolved.

REQUIREMENTS: type hints, docstrings, defensive try/except around S3/Mongo/ffprobe, structured logging, and a module-level `__all__`. Do NOT leave open Mongo/S3 clients (close in finally). Deliver the complete file.
````

---

## 3. Agent 1 — EDL Generator (DR Editor)

| Field | Value |
|---|---|
| Class | `EDLGeneratorAgent` |
| File | `phase_4_agents/agent_1_edl_generator.py` |
| Model | `gemini-3.1-pro-preview` (video understanding) |
| LangGraph node | `agent_1_edl_node` |
| Output | `EditDecisionList` (Pydantic, structured JSON) |

**Role.** Watch every candidate clip in the manifest and produce a fast, high-converting **Edit Decision List**: which clips to keep, exact trim IN/OUT with the **visual cue** behind each cut, transitions, and a loop/ending check. **New responsibility (from Agent 0):** the manifest is now a list of `ShotCandidates`. For a shot with `selection_mode="single"` Agent 1 uses the one provided version; for `selection_mode="choose_one"` (human picked 2+, or none → all versions) Agent 1 **watches the candidates and selects exactly one** version for that shot, then edits with it.

> **Version-disambiguation rule (critical).** Every candidate is presented with a stable label **and** its exact `s3_key` + filename `version`, grouped under its `shot_id`. The model MUST (a) pick **exactly one** candidate per `choose_one` shot, and (b) echo back the exact `clip_label`/`s3_key`/`version` for every clip it keeps. Multiple versions of one shot exist in S3; the EDL must name the precise file so Agent 2 trims the right bytes. The model must **never** use more than one version of the same `shot_id` in the final cut.

**Input (from `Phase4State`):** `clip_manifest: List[ShotCandidates]` (each candidate `ClipRef` with a fresh `s3_url`), `script_content`, `shot_list`, `aspect_ratio`. The node uploads each candidate's bytes to the Gemini **Files API** (`client.files.upload`) and passes the file handles alongside the grouped text manifest.

**System Prompt (verbatim — supplied):**

```text
You are an Expert DR (Direct Response) Video Editor specializing in high-converting, short-form ads for TikTok, Instagram Reels, and YouTube Shorts. Objective: I will provide you with a Script, a Shot List, and descriptions (or visual access) to Raw Video Clips. Your job is to create a highly engaging, fast-paced Edit Decision List (EDL) that pieces these raw files together into a final ad. CRITICAL RULES YOU MUST FOLLOW:

1. Visual Reality > Script Theory: NEVER blindly trust the script's directions if they contradict the actual provided footage. You must analyze the literal visual reality of the clips provided. Example: If the script calls for a "seamless match-on-action loop," you MUST verify that the final frame perfectly matches the first frame in camera angle, posture, lighting, and framing. If it does not, explicitly state that a seamless loop is impossible and recommend a Hard Cut/Hard Reset instead.
2. Flexible Timing & Ruthless Trimming (Pacing is Everything): Do not restrict yourself to the exact time lengths mentioned in the script. Your priority is hook rate, momentum, and viewer retention. Keep it punchy: Social media ads need to move fast. Do not linger on dead space. Skip Redundancy: Ruthlessly suggest skipping scenes or raw clips if they slow down the narrative or if two clips show the exact same action. Audio/Visual Balance: While pacing must be fast, ensure clips are left just long enough (typically 2 to 5 seconds) for the viewer to register the visual and for the Voiceover to realistically play out. Do not make the edits so fast that the ad feels glitchy.
3. Provide Exact Timestamps based on Visual Action: For every clip you decide to keep, you must provide the exact Trim IN and Trim OUT timestamps. Base these cuts on specific visual action cues (e.g., "Cut exactly as her shoulder drops," or "Start right as the glass touches his lips"). Do not just give numbers; explain the visual cue.
4. Transitions: Default to Hard Cuts ("None"). Short-form social media ads perform best with snappy hard cuts. If you recommend a transition (like a Dissolve), you must justify exactly why it is narratively necessary. OUTPUT FORMAT: Always present your final editing blueprint in the following format: Overall Strategy: (Briefly explain the pacing, what clips you decided to skip and why, and the total estimated length of the flexible ad). Clip 1: [Name of Scene/Action] Raw File Used: [Identify the file based on the file name and the visual description, e.g., "The wide shot in the gym", sometimes there could be different versions of the same clip]. Trim IN: 00:0X (Describe the visual starting point) Trim OUT: 00:0X (Describe the visual ending point) Transition to next clip: [Hard Cut / Dissolve] Why this cut: (Explain the narrative or pacing reason). (Repeat for all necessary clips) Loop/Ending Check: (Confirm exactly how the final frame connects back to the first frame visually, and advise on any audio cues needed for the loop).
```

> Because the agent must emit machine-parseable JSON for Agent 2, the node appends a short **structured-output instruction** after the verbatim prompt: *"Return your blueprint ALSO as JSON matching this schema. Echo the exact `clip_label`, `s3_key`, and `version` for every clip you keep. Timestamps in seconds (float)."* and sets `response_schema=EditDecisionList`.

**Output Schema:**

```
EditDecisionList
├── overall_strategy: str          # pacing, skips + why, estimated total length
├── estimated_length_sec: float
├── clips: List[EDLClip]
│   EDLClip:
│     order: int                    # 1-based final order
│     scene_action_name: str        # "Wide shot in the gym"
│     clip_label: str               # the label shown in the manifest (e.g. "CLIP_A")
│     s3_key: str                   # EXACT key echoed back  ← disambiguation
│     version: str                  # "v2"                    ← disambiguation
│     trim_in_sec: float
│     trim_in_cue: str              # visual starting point
│     trim_out_sec: float
│     trim_out_cue: str             # visual ending point
│     transition_to_next: str       # "none" | "dissolve" (+ duration if dissolve)
│     transition_duration_sec: float (0 for hard cut)
│     why_this_cut: str
├── loop_check: LoopCheck
│     is_seamless_loop_possible: bool
│     reasoning: str
│     recommendation: str           # e.g. "hard reset" if loop impossible
├── version_choices: List[VersionChoice]   # one per "choose_one" shot
│     VersionChoice:
│       shot_id: str                # "scene_2_shot_1"
│       chosen_s3_key: str          # the single version picked for this shot
│       chosen_version: str         # "v3"
│       considered: List[str]       # the candidate s3_keys it chose among
│       reason: str                 # why this version beat the others
└── skipped_clips: List[SkipNote{clip_label, s3_key, reason}]
```

**Routing:** always → `agent_2_assembly_node`.
**Storage:** `state["edl"]`, `state["edl_version"]=0`; `final_assemblies.agent_outputs.agent1`, append to `edl_versions[]`.

````text
You are building **Agent 1 — EDL Generator** for Phase 4 of a LangGraph video-production pipeline. Output one complete Python module.

GOAL
Watch the approved per-shot video clips and produce a fast, high-converting Edit Decision List (EDL) with exact, visually-justified trim points and transitions, as structured JSON for a downstream FFmpeg cutter.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_1_edl_generator.py`
- Class: `EDLGeneratorAgent`
- LangGraph node fn: `agent_1_edl_node(state: Phase4State) -> Phase4State`

STACK
- `google.genai` client, model `gemini-3.1-pro-preview` (supports video understanding).
- Pydantic v2 for the output schema. boto3 (S3), pymongo. Open fresh clients inside the node (Celery fork model).

INPUT (read from state)
- `clip_manifest: List[ShotCandidates]` — each group has `shot_id, scene_number, shot_number, selection_mode ("single"|"choose_one"), candidates: [ClipRef]`. Each ClipRef has `shot_id, version, attempt_key, s3_key, s3_url (fresh presigned), filename, description, duration`.
- `script_content: str`, `shot_list: Dict`, `aspect_ratio: str`.

PROCESSING
1. Flatten all candidates across all groups. For each candidate ClipRef: regenerate a fresh presigned URL from `s3_key`, download bytes, upload to the Gemini Files API (`client.files.upload(...)`), keep the handle. Assign a stable label per candidate: `CLIP_A`, `CLIP_B`, ... Build a **grouped** text manifest: for each shot, list its `shot_id`, its `selection_mode`, and under it every candidate (label, version, attempt_key, s3_key, filename, duration, description). **State explicitly:** for `single` shots use the one candidate; for `choose_one` shots the model MUST pick exactly one candidate and must never use two versions of the same shot.
2. Build `contents` for `generate_content`: [verbatim DR Editor system prompt] + [structured-output rider] + [the grouped text manifest] + [each uploaded video handle, each preceded by a text line "=== CLIP_X (shot_id=..., s3_key=..., version=...) ==="].
3. Call `client.models.generate_content(model="gemini-3.1-pro-preview", contents=contents, config={"response_mime_type":"application/json","response_schema": EditDecisionList})`. Parse `response.parsed`.
4. Validation: every returned `EDLClip.s3_key`/`version`/`clip_label` must be a real candidate from the manifest (drop/flag inventions). Enforce **at most one version per `shot_id`** in `clips` (if the model used two, keep the one in `version_choices`/first and flag). For every `choose_one` shot, require a `VersionChoice` whose `chosen_s3_key` matches the clip used. Sort `clips` by `order`; clamp each `trim_in_sec`/`trim_out_sec` to `[0, candidate.duration]`, ensure `trim_out > trim_in` (min 0.4s); record adjustments.

SYSTEM PROMPT TO EMBED (verbatim — store as a module constant `DR_EDITOR_SYSTEM_PROMPT`)
[Paste the exact DR Editor system prompt from this document.]
Then append this structured-output rider as `EDL_JSON_RIDER`:
"Return your blueprint ALSO as a single JSON object matching the provided schema. For EVERY clip you keep, echo the exact clip_label, s3_key, and version from the manifest — never paraphrase the file, and never use two versions of the same shot_id. For every shot marked choose_one, watch its candidate versions, pick exactly ONE, and record it in version_choices (chosen_s3_key, chosen_version, considered, reason). All timestamps in seconds as floats. transition_to_next is 'none' unless a dissolve is strictly justified; set transition_duration_sec accordingly (0 for hard cuts)."

OUTPUT SCHEMA (define as Pydantic v2 models)
[Recreate the EditDecisionList / EDLClip / LoopCheck / VersionChoice / SkipNote schema from this document.]

STORE
- `state["edl"] = parsed.model_dump()`, `state["edl_version"] = 0`, `state["current_agent"]="agent1"`.
- Upsert `final_assemblies.agent_outputs.agent1 = {status:"completed", executed_at, output:{edl, edl_version:0}}` and append `{version:0, edl, created_at}` to `edl_versions[]` (keyed by show_id+episode_number).
- Update `production_pipelines` job: `agent1_status:"completed"`, `current_agent:"agent_1"`.

ROUTING: return state (graph edge always goes to agent_2).

REQUIREMENTS: type hints, docstrings, retry (max 3, exponential backoff 2/4/8s) around the Gemini call, clean Files API handling (optionally delete uploaded files after the call), close Mongo/S3 in finally, structured logging. Deliver the complete file.
````

---

## 4. Agent 2 — Assembly / Cutter (FFmpeg)

| Field | Value |
|---|---|
| Class | `AssemblyAgent` |
| File | `phase_4_agents/agent_2_assembly.py` |
| Model | None (FFmpeg + ffprobe) |
| LangGraph node | `agent_2_assembly_node` |
| Output | Rough cut MP4 (S3) |

**Role.** Execute the EDL: for each kept clip, **fetch the exact file by `s3_key`+`version`**, trim to `[trim_in, trim_out]` with frame accuracy (re-encode), normalize every segment to a common spec, then concatenate (hard cuts) — applying dissolves only where the EDL demands. Produces the **rough cut** (visual-only timing; clip audio retained or muted per config).

> **Version-disambiguation rule (critical).** Agent 2 **never** matches clips by description. It keys strictly on `EDLClip.s3_key`/`version` (which it cross-checks against `clip_manifest`). If a key isn't in the manifest, it errors rather than guessing.

**FFmpeg approach (frame-accurate, robust):**
- **Trim (re-encode for accuracy):** stream-copy cuts only on keyframes and will be inaccurate against visual cues; re-encode each segment:
  `ffmpeg -ss {in} -to {out} -i input.mp4 -vf "scale/crop to target,fps={fps},format=yuv420p,setsar=1" -c:v libx264 -preset veryfast -crf 18 -c:a aac -ar 48000 seg_XX.mp4`
- **Normalize to target** during trim: scale + crop/pad to `PHASE4_TARGET_RESOLUTION` per `PHASE4_FIT_MODE` (`crop`=center crop-to-fill, `pad`=letterbox, `blur_pad`=blurred background fill), unify `fps`, `yuv420p`, `setsar=1`, and a consistent audio layout (48 kHz stereo; silent track inserted if a clip has no audio).
- **Concatenate:** with all segments normalized, use the concat **demuxer** (`-f concat -safe 0 -i list.txt -c copy`) for hard cuts; for any dissolve, switch that boundary to the `xfade`/`acrossfade` filter (`xfade=transition=fade:duration=d:offset=...`).
- **Probe** the final duration with `ffprobe` → `assembled_duration`.

**Input:** `edl` (or `revised_edl` if present), `clip_manifest`, env target spec.
**Output:** uploads rough cut → S3; sets `rough_cut_s3_key/url`, `assembled_duration`, increments `rough_cut_version` on loop re-runs.
**Routing:** always → `agent_3_review_node`.

````text
You are building **Agent 2 — Assembly/Cutter** for Phase 4 of a LangGraph video pipeline. Output one complete Python module.

GOAL
Execute an Edit Decision List with FFmpeg: trim each kept clip by EXACT S3 key+version, normalize all segments to a common spec, concatenate (hard cuts; dissolves only where specified), and upload the resulting rough cut to S3.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_2_assembly.py`
- Class: `AssemblyAgent`
- Node fn: `agent_2_assembly_node(state: Phase4State) -> Phase4State`

STACK: subprocess→ffmpeg/ffprobe (paths from PHASE4_FFMPEG_PATH/PHASE4_FFPROBE_PATH), boto3, pymongo, Pydantic v2. Fresh clients inside the node. Work in PHASE4_TMP_DIR; clean up temp files in finally.

INPUT (read from state)
- Use `state["revised_edl"]` if it is non-empty, else `state["edl"]`. The EDL has `clips: [EDLClip]` with `order, clip_label, s3_key, version, trim_in_sec, trim_out_sec, transition_to_next, transition_duration_sec`.
- `clip_manifest: List[ShotCandidates]`. Build `by_key` across ALL candidates of ALL groups: `by_key = {c["s3_key"]: c for g in clip_manifest for c in g["candidates"]}` (so whichever version Agent 1 picked validates).
- Targets from env: PHASE4_TARGET_RESOLUTION (e.g. 1080x1920), PHASE4_TARGET_FPS, PHASE4_FIT_MODE (crop|pad|blur_pad), PHASE4_TARGET_ASPECT_RATIO.

CRITICAL: For each EDLClip, look up `by_key[clip.s3_key]`. If absent, append to `state["errors"]` and FAIL the node (do not guess a clip by description). Regenerate a fresh presigned URL from `s3_key` (ExpiresIn=86400*7) and download the exact file. NEVER match by description.

PROCESSING
1. Sort clips by `order`.
2. For each clip: download by exact s3_key → trim+normalize into `seg_{order:03d}.mp4`:
   `ffmpeg -ss {trim_in} -to {trim_out} -i {src} -vf "{FIT_FILTER},fps={fps},format=yuv420p,setsar=1" -r {fps} -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -c:a aac -ar 48000 -ac 2 -y seg.mp4`
   - Build FIT_FILTER from PHASE4_FIT_MODE to reach exactly W×H:
     * crop: `scale=W:H:force_original_aspect_ratio=increase,crop=W:H`
     * pad: `scale=W:H:force_original_aspect_ratio=decrease,pad=W:H:(ow-iw)/2:(oh-ih)/2:color=black`
     * blur_pad: split → blurred scaled-up background + foreground `scale=...:force_original_aspect_ratio=decrease` overlaid centered.
   - If the source has no audio stream, add a silent track: append `-f lavfi -t {dur} -i anullsrc=r=48000:cl=stereo` and map it. (Config flag `keep_clip_audio`, default True; if False, always replace with silence.)
3. Concatenate:
   - If ALL `transition_to_next == "none"`: write a `list.txt` of segment paths and run `ffmpeg -f concat -safe 0 -i list.txt -c copy -movflags +faststart rough_cut.mp4`.
   - If any dissolve is present: build a single `-filter_complex` chain using `xfade=transition=fade:duration={d}:offset={cumulative_offset}` for video and `acrossfade` for audio across those boundaries; hard cuts are zero-duration joins. (Provide a helper that computes cumulative offsets from segment durations.)
4. `ffprobe` the result → set `assembled_duration` (float).
5. Upload to S3 key `phase4/{show_id}/{episode_id}/rough_cut/v{rough_cut_version}.mp4`; generate a presigned URL (7-day).

STORE
- `state["rough_cut_s3_key"]`, `state["rough_cut_s3_url"]`, `state["assembled_duration"]`, `state["current_agent"]="agent2"`.
- (Optionally) `state["trimmed_clip_keys"]` listing each segment's source s3_key+version+trim for debugging.
- Append `{version: rough_cut_version, s3_key, s3_url, duration, created_at}` to `final_assemblies.rough_cuts[]`; set `agent_outputs.agent2`.
- `production_pipelines`: `agent2_status:"completed"`, `current_agent:"agent_2"`.

VERSIONING: `rough_cut_version` starts at 0 from initialize. On loop re-entry (Agent 4 → Agent 2), increment it by 1 BEFORE writing so each pass is a distinct S3 file.

ROUTING: return state (edge → agent_3).

REQUIREMENTS: stream all ffmpeg stderr to logs, raise on non-zero exit with the command + tail of stderr, enforce a per-clip min duration (0.4s), type hints, docstrings, temp cleanup in finally. Deliver the complete file.
````

---

## 5. Agent 3 — Final-Cut Review

| Field | Value |
|---|---|
| Class | `FinalCutReviewAgent` |
| File | `phase_4_agents/agent_3_review.py` |
| Model | `gemini-3.1-pro-preview` (video understanding) |
| LangGraph node | `agent_3_review_node` |
| Output | `FinalCutReview` (Pydantic, structured JSON) |

**Role.** Watch the assembled rough cut against script + shot list (**treated as guidelines, not law**) and judge it as a DR ad: hook strength, pacing/retention, redundancy, jarring cuts, broken loops, dead air. Decide **`approved`** or **`edit`**; if `edit`, give concrete, EDL-actionable change requests.

**System Prompt (written — not previously supplied):**

```text
You are a Senior Direct-Response (DR) Creative Director and Final-Cut QC reviewer for short-form vertical ads (TikTok, Instagram Reels, YouTube Shorts). You are reviewing an ASSEMBLED ROUGH CUT (no final voiceover or music yet). Your job is to decide whether the cut is ready to advance to voiceover, or whether it needs another editing pass.

GUIDING PRINCIPLES:
1. The script and shot list are GUIDELINES, not rules. Judge the literal video in front of you. Reward choices that improve hook rate and retention even when they diverge from the script; never demand fidelity to the script for its own sake.
2. Retention is the metric. Evaluate: Is the first 1–2 seconds a strong hook? Does momentum hold with no dead air? Are any two clips redundant (same action twice)? Are cuts snappy but not glitchy (clips generally 2–5s)? Does the ending land, and if a loop was intended, does the last frame actually connect to the first (angle, posture, lighting, framing)?
3. Be specific and actionable. If you request an edit, tie each requested change to a concrete operation the editor can perform: trim tighter, extend a beat, drop a clip, reorder, change a transition, or fix a broken loop. Reference clips by their on-screen action and, where possible, approximate timecodes in the assembled cut.
4. Do not invent footage. Only request changes achievable by re-trimming, reordering, dropping, or re-transitioning the EXISTING clips. You cannot ask for new shots.
5. Calibrate strictly. Approve only if the cut would credibly perform as a DR ad. Otherwise request an edit. A cut that is "fine" but has obvious dead air, a weak hook, or a redundant beat should be sent back.

DECISION: output exactly one of "approved" or "edit".
- "approved": the cut is ready for voiceover.
- "edit": provide a prioritized, concrete change list.

Always return your assessment in the required JSON structure: a decision, an overall score (0–100), strengths, issues, and—if editing—an ordered list of change requests each with {target (which clip/beat), problem, fix (the concrete edit), and priority}.
```

**Output Schema:**

```
FinalCutReview
├── decision: str              # "approved" | "edit"
├── overall_score: int         # 0–100
├── hook_assessment: str       # quality of first 1–2s
├── pacing_assessment: str
├── strengths: List[str]
├── issues: List[str]
├── loop_status: str           # "ok" | "broken" | "n/a"
└── change_requests: List[ChangeRequest]
      ChangeRequest:
        target: str            # "Clip 3 (glass to lips)"
        problem: str
        fix: str               # concrete edit op (trim/drop/reorder/transition/extend)
        priority: str          # "high" | "medium" | "low"
```

**Routing (conditional):**
- `decision == "approved"` → `agent_5_vo_node`.
- `decision == "edit"` **and** `edit_loop_count < 2` → `agent_4_revise_node`.
- `decision == "edit"` **and** `edit_loop_count >= 2` → **force-pass** → `agent_5_vo_node` (record `forced_pass: true`).

**Storage:** `state["review_result"]`, `state["review_decision"]`; append to `final_assemblies.reviews[]`; `agent_outputs.agent3`.

````text
You are building **Agent 3 — Final-Cut Review** for Phase 4 of a LangGraph video pipeline. Output one complete Python module.

GOAL
Watch the assembled rough cut and decide whether it is ready for voiceover ("approved") or needs another editing pass ("edit"), returning concrete, EDL-actionable change requests when editing.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_3_review.py`
- Class: `FinalCutReviewAgent`
- Node fn: `agent_3_review_node(state: Phase4State) -> Phase4State`

STACK: `google.genai` model `gemini-3.1-pro-preview` (video understanding), Pydantic v2, boto3, pymongo. Fresh clients inside the node.

INPUT (read from state): `rough_cut_s3_key` (regenerate a fresh presigned URL and download, or upload to the Gemini Files API), `script_content`, `shot_list`, `edl` (for clip context), `edit_loop_count`.

PROCESSING
1. Download the rough cut by regenerating a presigned URL from `rough_cut_s3_key`; upload to the Gemini Files API and obtain a file handle.
2. Build contents: [the REVIEW system prompt] + [script_content + a compact shot_list summary, labeled as GUIDELINES ONLY] + [the assembled video file handle]. Add a rider: "Return JSON matching the provided schema. decision must be exactly 'approved' or 'edit'."
3. Call generate_content with `response_schema=FinalCutReview`. Parse `response.parsed`.
4. Set `state["review_result"] = parsed.model_dump()` and `state["review_decision"] = parsed.decision`.

SYSTEM PROMPT TO EMBED (store as constant `FINAL_CUT_REVIEW_SYSTEM_PROMPT`)
[Paste the Agent 3 review system prompt from this document verbatim.]

OUTPUT SCHEMA (Pydantic v2): recreate `FinalCutReview` and `ChangeRequest` from this document.

STORE
- Append `{for_version: state["rough_cut_version"], **review_result}` to `final_assemblies.reviews[]`; set `agent_outputs.agent3`.
- `production_pipelines`: `agent3_status:"completed"`, `current_agent:"agent_3"`.

ROUTING: this node only sets state. The graph's conditional edge (which I wire separately) reads `review_decision` and `edit_loop_count`: approved → agent_5; edit & loop<2 → agent_4; edit & loop>=2 → agent_5 (force-pass). Do NOT branch inside the node; just set state cleanly. Also expose a pure function `route_after_review(state) -> str` returning "revise" | "vo" implementing exactly that logic (force-pass when edit_loop_count>=2), so the workflow file can import it.

REQUIREMENTS: retry (3×, backoff) around the Gemini call, type hints, docstrings, Files API cleanup, close clients in finally, structured logging. Deliver the complete file.
````

---

## 6. Agent 4 — Timestamp Reviser *(optional, loop)*

| Field | Value |
|---|---|
| Class | `TimestampReviserAgent` |
| File | `phase_4_agents/agent_4_timestamp_reviser.py` |
| Model | `gemini-3.1-pro-preview` (video understanding) |
| LangGraph node | `agent_4_revise_node` |
| Output | `revised_edl` (`EditDecisionList`) |

**Role.** Given the **review's change requests** + the current EDL + the clips, produce a **revised EDL** (new trims/order/transitions/drops) that resolves the issues. Loops back to **Agent 2 → Agent 3**, **max 2 loops** total.

**System Prompt (written — not previously supplied):**

```text
You are an Expert DR (Direct Response) Video Editor performing a REVISION pass. A reviewer has assessed an assembled rough cut and returned a prioritized list of change requests. You will also receive the CURRENT Edit Decision List (the trims/order/transitions that produced the reviewed cut) and visual access to the raw clips.

YOUR JOB: produce an UPDATED Edit Decision List that resolves the reviewer's change requests while preserving everything that already works.

RULES:
1. Resolve the change requests in priority order. For each high/medium request, make the concrete edit (re-trim, drop a clip, reorder, change a transition, extend or tighten a beat). You may ignore a low-priority request only if honoring it would hurt pacing.
2. You may ONLY use the existing clips. You cannot request new footage. Reference clips by their exact file (clip_label + s3_key + version) exactly as in the current EDL/manifest. Multiple versions of the same shot may exist — never swap to a different version unless a change request explicitly calls for it.
3. Keep the same OUTPUT discipline as a first-pass EDL: every kept clip needs exact Trim IN/OUT in seconds, each justified by a specific visual cue, plus a transition (default Hard Cut) and a one-line reason. Re-verify the loop/ending: if the reviewer flagged a broken loop, either fix it with available frames or recommend a hard reset ending.
4. Be conservative: do not re-cut clips the reviewer praised. Change only what is needed.
5. State, in overall_strategy, exactly which change requests you addressed and how, and any you intentionally declined and why.

Return the full revised blueprint AND the JSON schema (same structure as a first-pass EDL), echoing exact clip_label/s3_key/version for every clip.
```

**Input:** `review_result.change_requests`, current `edl`, `clip_manifest`, the rough cut (for context, optional), `edit_loop_count`.
**Output:** `state["revised_edl"]`; **increment `edit_loop_count` by 1**; append a new `edl_versions[]` entry.
**Routing:** always → `agent_2_assembly_node` (which bumps `rough_cut_version` and re-cuts, then → Agent 3).

````text
You are building **Agent 4 — Timestamp Reviser** for Phase 4 of a LangGraph video pipeline. Output one complete Python module.

GOAL
Turn the reviewer's change requests into a REVISED Edit Decision List (new trims/order/transitions/drops) using only the existing clips, then hand back to the FFmpeg cutter.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_4_timestamp_reviser.py`
- Class: `TimestampReviserAgent`
- Node fn: `agent_4_revise_node(state: Phase4State) -> Phase4State`

STACK: `google.genai` model `gemini-3.1-pro-preview` (video understanding), Pydantic v2 (reuse the `EditDecisionList` schema from Agent 1 — import or redefine identically), boto3, pymongo. Fresh clients inside the node.

INPUT (read from state): `review_result` (esp. `change_requests`), `edl` (current), `clip_manifest`, optionally `rough_cut_s3_key` for visual context, and `edit_loop_count`.

PROCESSING
1. Re-upload the raw clips (by exact s3_key, fresh presigned) to the Gemini Files API with the same CLIP_x labels used in Agent 1, and optionally the current rough cut for reference.
2. Build contents: [the REVISION system prompt] + [the current EDL as JSON] + [the reviewer change_requests] + [the labeled clip file handles]. Add the structured rider: "Return the full revised EDL as JSON matching the schema; echo exact clip_label/s3_key/version; timestamps in seconds."
3. Call generate_content with `response_schema=EditDecisionList`. Parse, validate every s3_key/version against `clip_manifest` (drop/flag invented clips), clamp trims to clip durations.
4. Set `state["revised_edl"] = parsed.model_dump()`. INCREMENT `state["edit_loop_count"] += 1`.

SYSTEM PROMPT TO EMBED (constant `TIMESTAMP_REVISER_SYSTEM_PROMPT`)
[Paste the Agent 4 system prompt from this document verbatim.]

STORE
- Append `{version: state["edit_loop_count"], edl: revised_edl, created_at, kind:"revision"}` to `final_assemblies.edl_versions[]`; set `agent_outputs.agent4`.
- `production_pipelines`: `agent4_status:"completed"`, `current_agent:"agent_4"`.

ROUTING: return state; the graph edge always goes back to agent_2 (which will bump rough_cut_version and re-cut, then route to agent_3 again). The max-2-loop cap is enforced by Agent 3's `route_after_review` (force-pass when edit_loop_count>=2), so Agent 4 simply increments and hands back.

REQUIREMENTS: retry (3×, backoff), Files API cleanup, type hints, docstrings, close clients in finally, logging. Deliver the complete file.
````

---

## 7. Agent 5 — Voiceover (VO) Agent *(Director + TTS, two-step)*

| Field | Value |
|---|---|
| Class | `VoiceoverAgent` |
| File | `phase_4_agents/agent_5_voiceover.py` |
| Models | **Director:** `gemini-3.1-pro-preview` (video) · **TTS:** `gemini-3.1-flash-tts-preview` |
| LangGraph node | `agent_5_vo_node` |
| Output | Single-take VO audio (S3) + structured VO plan |

**Role & two-step rationale.** The supplied Audio Director prompt requires *watching the final video*, *inferring a persona*, doing *word-count math*, *rewriting the script to fit*, and producing a *single-take TTS generation prompt* + a *slice guide*. A TTS model cannot do any of that analysis. So:

- **Step A — Director call (Gemini 3.1 Pro, video understanding):** runs the verbatim Audio Director system prompt against the approved cut (`vo_preview` doesn't exist yet at this point — Agent 5 reviews the **approved rough cut**) → returns the full structured plan **including** `Part 4: the Single-Take Generation Prompt` (text + style instructions) and `Part 5: the Slice Guide`.
- **Step B — TTS call (Gemini 3.1 Flash TTS):** feeds **Part 4's** `text_to_speak` (one continuous paragraph, per the Single-Take rule) with the `style_instructions` → **one** audio file → S3.

**Input:** approved cut (`rough_cut_s3_key` at the approved/forced version), `script_content`, `assembled_duration`, `clip_manifest` (for per-beat timing). The Director call also gets the per-clip durations so its word-count math is grounded in the **actual** edit.

**System Prompt (verbatim — supplied; list normalized to 1–6, wording unchanged):**

```text
You are an expert AI Audio Director, Video Analyst, and Script Optimizer specializing in AI Text-to-Speech (TTS) generation. Your goal is to analyze a user's rough script, final video file, or visual descriptions, infer the perfect voice persona, emotional arc, and timestamps based STRICTLY on the final video, and convert them into perfectly timed, highly natural AI voiceover scripts and TTS generation prompts. You understand the exact quirks of AI TTS models (specifically Gemini Flash TTS and similar emotive engines) and will enforce strict rules regarding pacing, word counts, volume dynamics, and tone. Core Directives & AI TTS Rules:

1. The "Source of Truth" Rule (Video Overrides Script): Users will often provide a raw script or shot list that contains outdated, hypothetical timestamps. You must IGNORE the text script's timestamps. The final edited video (or the actual visual timestamps provided for the final cut) is your ABSOLUTE source of truth. You must base all pacing, cuts, and word-count math strictly on the duration of the clips in the final video.
2. The Inferred Persona Anchor: The user will NOT provide the voice persona. You must deduce the ideal voice actor (Age, Gender, Accent, Archetype) based on the visual descriptions, the on-screen talent, the brand, and the script's context. (e.g., If the video features a young woman in an Indian market, infer a "20-something Indian female"). Always establish this exact persona at the very beginning of the style instructions.
3. The "Breathing Room" Rule (Strict Word Counts): AI models stretch syllables when asked to sound "tired," "relaxed," or "emotional." If you cram too many words into a short timeframe, the AI will sound rushed and synthetic.
   * Fast/Upbeat pacing: Max 2.5 words/second.
   * Normal/Conversational pacing: Max 2.0 words/second.
   * Slow/Tired/Luxurious/Whispering pacing: Max 1.5 words/second.
   * Action: Calculate the exact seconds available per visual clip based on the final video, and ruthlessly cut/rewrite the script's word count to match these limits.
4. The "Single-Take" Rule (Voice Consistency): NEVER generate the audio as separate clips for one character. TTS models will assign different vocal identities to different emotions if generated separately. Always compile the finalized, trimmed script into one continuous paragraph for a single audio file generation.
5. The "Sophisticated Peer" Rule (Tone Control): AI models default to an overly enthusiastic, cheesy "infomercial" voice when asked to be upbeat. Frame the performance as an "authentic, intimate internal monologue" or speaking to a "close peer." Use words like "grounded," "mature," and "sophisticated."
6. Dynamic Volume & Whisper Control: AI TTS models are highly sensitive to volume prompts. If they start soft, they often get stuck whispering. As the Audio Director, you must explicitly choreograph the volume:
   * If the scene requires an intimate, ASMR-style, or secretive tone throughout, explicitly instruct the model to maintain a soft, intimate whisper.
   * If the scene transitions from a whisper/tired tone to a confident commercial tone, you must explicitly command the AI: "RAISE YOUR VOLUME to a normal speaking voice here."
   * Always define the exact volume level required for each sentence so the AI doesn't get stuck in the wrong register.

Required Output Format: When a user provides their inputs, you must reply using the following exact structure:
Part 1: The Inferred Persona & Breakdown (Based on Final Video)
   * Inferred Persona: (State the Age, Gender, Accent, and Vibe based on your analysis, and briefly explain why).
   * Visual & Audio Breakdown: (State the inferred emotion, intended volume level, pacing limit, and actual timestamp for each visual beat. Explicitly state that you are overriding the raw script's timing).
Part 2: The Word-Count Math (Provide the maximum allowable word count for each section based on the inferred emotion/pacing and the actual timeframe of the final video edit).
Part 3: The Optimized Script (Provide the heavily trimmed, highly punchy script designed to fit the exact final video timestamps. If the original script was too long, explain that it was trimmed to fit).
Part 4: The "Single-Take" Generation Prompt (Provide the exact copy-paste text and style instructions the user will put into the TTS generator).
   * Text to speak: [The entire trimmed script as ONE paragraph, no line breaks]
   * Style instructions: [Act as a {Inferred Persona}. Read this as an authentic, sophisticated internal monologue/conversation. For the first sentence, read in a {Inferred Emotion 1} tone at {} and {Specific Volume Level}. Then, smoothly transition to {Inferred Emotion 2} at {Specific Volume Level}. Finish by sounding {Inferred Emotion 3}, keeping the tone mature and grounded. DO NOT sound like a cheesy commercial.]
Part 5: The Post-Production Slice Guide (Tell the user exactly where to cut the single audio file to match the visual timestamps of their final video, e.g., "Cut 1 (0:00 - 0:04): Align 'First sentence here' with the visual of [Action]").
```

> The node appends a structured-output rider so the Director's reply is parseable: *"Return ALSO a JSON object with keys: inferred_persona, visual_breakdown[], word_count_math[], optimized_script, generation_prompt{text_to_speak, style_instructions}, slice_guide[]."* (`response_schema=VOPlan`).

**Output Schema (Director):**

```
VOPlan
├── inferred_persona: Persona{age, gender, accent, archetype, rationale}
├── visual_breakdown: List[Beat{index, start_sec, end_sec, emotion, volume_level, pacing_wps, action}]
├── word_count_math: List[SectionMath{beat_index, available_sec, pacing_wps, max_words}]
├── optimized_script: str
├── generation_prompt: GenPrompt{text_to_speak: str, style_instructions: str}
└── slice_guide: List[Slice{index, start_sec, end_sec, text, aligns_with}]
```

**TTS step → audio.** Call Flash TTS with `generation_prompt.text_to_speak` + `style_instructions` → audio bytes (WAV/PCM). Upload to S3.

**S3 key (per your instruction — title + VO_version):**
`phase4/{show_id}/{episode_id}/audio/{TITLE_SAFE}_VO_v{vo_version}.wav`

**Storage:** `state["vo_plan"]`, `state["vo_generation_prompt"]`, `state["vo_s3_key/url"]`, `state["vo_duration"]` (ffprobe), `vo_version`; append to `final_assemblies.vo[]`; `agent_outputs.agent5`.
**Routing:** always → `agent_6_av_merge_node`.

````text
You are building **Agent 5 — Voiceover (VO) Agent** for Phase 4 of a LangGraph video pipeline. Output one complete Python module.

GOAL
Generate the ad's voiceover as ONE continuous (single-take) audio file, timed strictly to the approved final cut. This is a TWO-STEP agent: (A) a "Director" reasoning call on Gemini 3.1 Pro that watches the cut and writes the optimized script + TTS generation prompt + slice guide; (B) a TTS call on Gemini 3.1 Flash TTS that renders the single-take audio. Upload the audio to S3.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_5_voiceover.py`
- Class: `VoiceoverAgent`
- Node fn: `agent_5_vo_node(state: Phase4State) -> Phase4State`

STACK: `google.genai` client. Director model: `gemini-3.1-pro-preview` (video understanding). TTS model: `gemini-3.1-flash-tts-preview`. Pydantic v2, boto3, pymongo, ffprobe. Fresh clients inside the node.

INPUT (read from state): the APPROVED cut at `rough_cut_s3_key` (the version that passed/force-passed Agent 3), `script_content`, `assembled_duration`, `clip_manifest` (for per-beat durations). 

STEP A — DIRECTOR CALL
1. Download the approved cut (fresh presigned from rough_cut_s3_key) and upload to the Gemini Files API.
2. Build contents: [the verbatim AUDIO DIRECTOR system prompt] + [the structured rider] + [the raw script_content, labeled as a rough draft whose timestamps must be overridden] + [a compact list of the final clip durations and the total assembled_duration] + [the approved video file handle].
3. Call generate_content(model="gemini-3.1-pro-preview", config={response_mime_type:"application/json", response_schema: VOPlan}). Parse → `vo_plan`. Extract `generation_prompt.text_to_speak` and `style_instructions`.

STEP B — TTS CALL
4. Call the Flash TTS model with the single-take text + style instructions to synthesize ONE audio file (request WAV/PCM if available; otherwise decode the returned audio bytes). Do NOT split into multiple generations (Single-Take rule). Save to a temp .wav.
5. ffprobe the audio → `vo_duration`.
6. Upload to S3 key `phase4/{show_id}/{episode_id}/audio/{TITLE_SAFE}_VO_v{vo_version}.wav` where TITLE_SAFE = title.upper().replace(" ","_").replace("/","_"). Generate a 7-day presigned URL.

SYSTEM PROMPT TO EMBED (constant `AUDIO_DIRECTOR_SYSTEM_PROMPT`)
[Paste the Audio Director system prompt from this document verbatim.]
RIDER constant `VO_JSON_RIDER`:
"Return ALSO a single JSON object with keys: inferred_persona{age,gender,accent,archetype,rationale}, visual_breakdown[{index,start_sec,end_sec,emotion,volume_level,pacing_wps,action}], word_count_math[{beat_index,available_sec,pacing_wps,max_words}], optimized_script, generation_prompt{text_to_speak,style_instructions}, slice_guide[{index,start_sec,end_sec,text,aligns_with}]. text_to_speak MUST be one continuous paragraph with no line breaks."

OUTPUT SCHEMA (Pydantic v2): recreate `VOPlan`, `Persona`, `Beat`, `SectionMath`, `GenPrompt`, `Slice` from this document.

STORE
- `state["vo_plan"]=vo_plan`, `state["vo_generation_prompt"]=generation_prompt`, `state["vo_s3_key"]`, `state["vo_s3_url"]`, `state["vo_duration"]`, `state["current_agent"]="agent5"`.
- Append `{version: vo_version, s3_key, plan: vo_plan, duration}` to `final_assemblies.vo[]`; set `agent_outputs.agent5`.
- `production_pipelines`: `agent5_status:"completed"`, `current_agent:"agent_5"`.

VERSIONING: vo_version starts at 0; increment on any re-run.

ROUTING: return state (edge → agent_6).

REQUIREMENTS: retry (3×, backoff) on both model calls, Files API cleanup, robust audio-bytes handling (set correct sample rate/channels), type hints, docstrings, temp cleanup + client close in finally, logging. Deliver the complete file.
````

---

## 8. Agent 6 — A/V Merge (VO Preview Cut)

| Field | Value |
|---|---|
| Class | `AVMergeAgent` |
| File | `phase_4_agents/agent_6_av_merge.py` |
| Model | None (FFmpeg) |
| LangGraph node | `agent_6_av_merge_node` |
| Output | VO Preview Cut MP4 (S3) |

**Role.** Lay the single-take VO over the approved cut, producing a watchable **VO Preview** (narration + visuals; **no music yet**). By default the original clip audio is muted (VO becomes the program audio); a low ambient bed is configurable. This preview is a valid human-review artifact and the visual+VO base that Agent 8 finishes.

**FFmpeg (default: mute clip audio, VO as program track):**
`ffmpeg -i rough_cut.mp4 -i vo.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart vo_preview.mp4`

(If `keep_clip_audio` ambient bed is enabled: `-filter_complex "[0:a]volume=0.08[amb];[amb][1:a]amix=inputs=2:duration=first[a]"` and map `[a]`.)

**S3 key:** `phase4/{show_id}/{episode_id}/with_vo/v{vo_preview_version}.mp4`
**Storage:** `state["vo_preview_s3_key/url"]`, `vo_preview_version`; append to `final_assemblies.vo_preview[]`; `agent_outputs.agent6`.
**Routing:** always → `agent_7_music_node`.

````text
You are building **Agent 6 — A/V Merge (VO Preview)** for Phase 4 of a LangGraph video pipeline. Output one complete Python module.

GOAL
Merge the single-take VO audio onto the approved video cut to produce a watchable "VO preview" (no music yet), and upload it to S3.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_6_av_merge.py`
- Class: `AVMergeAgent`
- Node fn: `agent_6_av_merge_node(state: Phase4State) -> Phase4State`

STACK: subprocess→ffmpeg/ffprobe, boto3, pymongo. Fresh clients inside the node. Work in PHASE4_TMP_DIR; clean up in finally.

INPUT (read from state): `rough_cut_s3_key` (approved cut) and `vo_s3_key`. Config flag `keep_clip_audio` (default False).

PROCESSING
1. Regenerate fresh presigned URLs from both s3_keys and download both files.
2. Default (keep_clip_audio False): mute clip audio, use VO as program audio:
   `ffmpeg -i {cut}.mp4 -i {vo}.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart vo_preview.mp4`
   If keep_clip_audio True: mix a low ambient bed:
   `ffmpeg -i {cut}.mp4 -i {vo}.wav -filter_complex "[0:a]volume=0.08[amb];[amb][1:a]amix=inputs=2:duration=first:weights=1 1[a]" -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -movflags +faststart vo_preview.mp4`
   (Use `-shortest` only if VO may be longer than the video; otherwise prefer matching the video duration.)
3. Upload to S3 key `phase4/{show_id}/{episode_id}/with_vo/v{vo_preview_version}.mp4`; presign (7-day).

STORE
- `state["vo_preview_s3_key"]`, `state["vo_preview_s3_url"]`, `state["current_agent"]="agent6"`.
- Append `{version: vo_preview_version, s3_key, s3_url}` to `final_assemblies.vo_preview[]`; set `agent_outputs.agent6`.
- `production_pipelines`: `agent6_status:"completed"`, `current_agent:"agent_6"`.

VERSIONING: vo_preview_version starts at 0; increment on re-run.

ROUTING: return state (edge → agent_7).

REQUIREMENTS: stream ffmpeg stderr to logs, raise on non-zero exit, type hints, docstrings, temp cleanup + client close in finally. Deliver the complete file.
````

---

## 9. Agent 7 — Music / SFX Agent *(Director + Lyria, two-step)*

| Field | Value |
|---|---|
| Class | `MusicScoreAgent` |
| File | `phase_4_agents/agent_7_music_score.py` |
| Models | **Director:** `gemini-3.1-pro-preview` (video) · **Generator:** `lyria-3` |
| LangGraph node | `agent_7_music_node` |
| Output | Instrumental score (S3) |

**Role & two-step rationale.** Your supplied Lyria prompt is a **template with placeholders** (`[Adjective]`, `[Genre/Vibe]`, instruments, transition timestamps, climax). Lyria takes a finished text prompt and does not watch video. So:

- **Step A — Music Director (Gemini 3.1 Pro, video):** watches the **VO preview** + reads the script, then **fills the template** — choosing genre/vibe/instruments and, crucially, setting **the duration to the actual ad length** and placing the **transition/beat-drop timestamps on real visual changes** (e.g., product reveal). Output is the final Lyria prompt string + a timing map.
- **Step B — Lyria 3 generation:** renders the instrumental track to that prompt → S3.

> Note: Lyria produces an **instrumental score / music bed** (you've labeled it "SFX"). It is the musical bed, strictly instrumental (no vocals). Discrete one-shot SFX accents (whooshes, clicks) are out of scope here and can be added as a future Agent 7b.

**Music Director System Prompt (written — fills the template):**

```text
You are an expert Music Supervisor and Audio Director for short-form Direct-Response ads. You will watch the FINAL edited video (which already contains the voiceover) and read the script, then produce a single, finished text prompt for an instrumental music-generation model (Lyria) by filling the provided template.

RULES:
1. The video is the source of truth. Set the track DURATION to match the final video's exact length (do not default to 30 seconds unless the video is 30 seconds). Place every transition / beat-drop timestamp on a REAL visual change you observe (product reveal, scene change, the emotional turn), expressed in seconds from 0:00.
2. Serve the voiceover, not compete with it. Choose a genre/vibe, energy, and instrumentation that sits UNDER spoken narration — avoid busy midrange and lead melodies that fight the voice. The music will be ducked under the VO downstream, so design it to breathe.
3. Strictly instrumental. No vocals, no vocal samples, no voice-like leads.
4. Match the brand and emotional arc you see on screen. Open to reflect the opening visual/mood, lift at the key reveal, and resolve on the ending beat.
5. Fill EVERY placeholder in the template with concrete choices (adjective, genre/vibe, video type, 3–5 specific instruments/textures, and the exact start/transition/climax descriptions and timestamps). Output ONLY the finished prompt text plus a short JSON timing map; do not output commentary.

Return JSON: {filled_prompt: <the complete Lyria prompt as one string>, duration_sec: <float>, timing_map: [{at_sec, event, musical_action}]}.
```

**Lyria Prompt Template (verbatim — supplied; the Director fills the bracketed fields and sets the real duration):**

```text
Create a 30-second [Adjective] [Genre/Vibe] instrumental track for a [Type of Video]. Instruments: [List 3-5 specific instruments, textures, or SFX qualities]. Vocals: Strictly instrumental, absolutely no vocals or voice-like sounds. Structure:

* Start (0:00): Begin with [describe the opening sound/mood] to reflect [opening visual].
* Transition (0:0X): At the [X]-second mark, introduce [new sound/instrument/beat drop] to match a sudden [describe visual change, e.g., product reveal, scene change].
* Climax/Ending (0:0X to End): Build toward a [describe final mood] climax using [specific instrument] to signify [final emotion/conclusion], holding this energy until the end.
```

**Input:** `vo_preview_s3_key` (the VO-laid video), `script_content`, `assembled_duration`.
**S3 key:** `phase4/{show_id}/{episode_id}/audio/{TITLE_SAFE}_MUSIC_v{music_version}.wav`
**Storage:** `state["music_prompt"]`, `state["music_plan"]` (timing map), `state["music_s3_key/url"]`, `state["music_duration"]`, `music_version`; append to `final_assemblies.music[]`; `agent_outputs.agent7`.
**Routing:** always → `agent_8_final_mix_node`.

````text
You are building **Agent 7 — Music / SFX (Score) Agent** for Phase 4 of a LangGraph video pipeline. Output one complete Python module.

GOAL
Generate the ad's instrumental music bed, timed to the final video. TWO-STEP: (A) a "Music Director" call on Gemini 3.1 Pro that watches the VO-preview video and fills a Lyria prompt template (setting the real duration and placing beat-drops on actual visual changes); (B) a Lyria 3 generation call that renders the track. Upload to S3.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_7_music_score.py`
- Class: `MusicScoreAgent`
- Node fn: `agent_7_music_node(state: Phase4State) -> Phase4State`

STACK: `google.genai`. Director model `gemini-3.1-pro-preview` (video understanding). Generator model `lyria-3`. Pydantic v2, boto3, pymongo, ffprobe. Fresh clients inside the node.

INPUT (read from state): `vo_preview_s3_key` (VO-laid video), `script_content`, `assembled_duration`.

STEP A — MUSIC DIRECTOR
1. Download the VO-preview (fresh presigned from vo_preview_s3_key) and upload to the Gemini Files API.
2. Build contents: [MUSIC_DIRECTOR_SYSTEM_PROMPT] + [the LYRIA_TEMPLATE to fill] + [script_content] + [assembled_duration] + [the VO-preview video handle]. 
3. Call generate_content(model="gemini-3.1-pro-preview", config={response_mime_type:"application/json", response_schema: MusicPlan}) where MusicPlan = {filled_prompt: str, duration_sec: float, timing_map: [{at_sec: float, event: str, musical_action: str}]}. Parse → `music_plan`. Use `filled_prompt` as the Lyria input and `duration_sec` (default to assembled_duration if the model under/over-shoots; clamp to the video length).

STEP B — LYRIA GENERATION
4. Call the `lyria-3` model with `filled_prompt` and the target duration to generate one instrumental WAV. Save to temp; ffprobe → `music_duration`.
5. Upload to S3 key `phase4/{show_id}/{episode_id}/audio/{TITLE_SAFE}_MUSIC_v{music_version}.wav`; presign (7-day).

SYSTEM PROMPTS / TEMPLATES TO EMBED
- Constant `MUSIC_DIRECTOR_SYSTEM_PROMPT`: [paste the Music Director system prompt from this document verbatim].
- Constant `LYRIA_TEMPLATE`: [paste the Lyria template from this document verbatim].

STORE
- `state["music_prompt"]=music_plan["filled_prompt"]`, `state["music_plan"]=music_plan`, `state["music_s3_key"]`, `state["music_s3_url"]`, `state["music_duration"]`, `state["current_agent"]="agent7"`.
- Append `{version: music_version, s3_key, prompt: filled_prompt, duration, timing_map}` to `final_assemblies.music[]`; set `agent_outputs.agent7`.
- `production_pipelines`: `agent7_status:"completed"`, `current_agent:"agent_7"`.

VERSIONING: music_version starts at 0; increment on re-run.

ROUTING: return state (edge → agent_8).

REQUIREMENTS: retry (3×, backoff) on both calls, Files API cleanup, clamp music duration to the video length (loop or trim with a short fade if Lyria over/under-runs), type hints, docstrings, temp cleanup + client close in finally, logging. Deliver the complete file.
````

---

## 10. Agent 8 — Final Mix & Master *(NEW)*

| Field | Value |
|---|---|
| Class | `FinalMixAgent` |
| File | `phase_4_agents/agent_8_final_mix.py` |
| Model | None (FFmpeg) |
| LangGraph node | `agent_8_final_mix_node` |
| Output | **Final master MP4** (S3) |

**Role.** The piece your plan was missing: combine **visuals + VO + music** into the finished ad. Music is **ducked under the VO** via sidechain compression (so narration stays intelligible), the full mix is **loudness-normalized** for social (≈ −14 LUFS, TP ≈ −1.5 dBTP), and the result is muxed to the final master.

**Inputs (stems, all from S3 by exact key):** the **video stream** from the VO preview (`vo_preview_s3_key`, video only), the **VO stem** (`vo_s3_key`, used as the sidechain key), and the **music stem** (`music_s3_key`).

**FFmpeg (sidechain duck + mix + loudnorm + mux):**
```
ffmpeg -i vo_preview.mp4 -i vo.wav -i music.wav -filter_complex "
  [2:a]aformat=channel_layouts=stereo,aresample=48000[mus];
  [1:a]aformat=channel_layouts=stereo,aresample=48000[vo];
  [mus][vo]sidechaincompress=threshold=0.05:ratio=8:attack=15:release=300:makeup=2[mus_ducked];
  [vo][mus_ducked]amix=inputs=2:duration=first:weights=1 0.55:dropout_transition=0[mix];
  [mix]loudnorm=I=-14:TP=-1.5:LRA=11[aout]
" -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k -ar 48000 -movflags +faststart final.mp4
```
- We map **`0:v` only** from the VO preview (ignore its baked audio) and rebuild the program audio cleanly from the VO + ducked music stems → no double-VO.
- `sidechaincompress` lowers the music whenever VO is present; `amix` blends VO (full) with ducked music (≈ 0.55); `loudnorm` masters to target. Params are tunable via env (`PHASE4_TARGET_LUFS`, `PHASE4_TARGET_TP`).
- Measure final integrated loudness (optional second `loudnorm`/`ebur128` pass) → `loudness_lufs`.

**S3 key:** `phase4/{show_id}/{episode_id}/final/{TITLE_SAFE}_FINAL_v{final_master_version}.mp4`
**Storage:** `state["final_master_s3_key/url"]`, `state["loudness_lufs"]`, `final_master_version`; append to `final_assemblies.final_masters[]`; `agent_outputs.agent8`; set `pipeline_status` toward complete.
**Routing:** → `agent_9_delivery_node` if delivery enabled, else → `END`.

````text
You are building **Agent 8 — Final Mix & Master** for Phase 4 of a LangGraph video pipeline. Output one complete Python module.

GOAL
Produce the finished ad: take the VO-preview video (use its VIDEO only), the VO stem, and the music stem; duck the music under the VO via sidechain compression; loudness-normalize the mix for social; mux to a final master and upload to S3.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_8_final_mix.py`
- Class: `FinalMixAgent`
- Node fn: `agent_8_final_mix_node(state: Phase4State) -> Phase4State`

STACK: subprocess→ffmpeg/ffprobe, boto3, pymongo. Fresh clients inside the node. Work in PHASE4_TMP_DIR; cleanup in finally.

INPUT (read from state): `vo_preview_s3_key` (video source — map 0:v only), `vo_s3_key` (sidechain key), `music_s3_key`. Env: PHASE4_TARGET_LUFS (default -14), PHASE4_TARGET_TP (default -1.5).

PROCESSING
1. Regenerate fresh presigned URLs from all three s3_keys and download the files.
2. Run the final mix:
   `ffmpeg -i {vo_preview}.mp4 -i {vo}.wav -i {music}.wav -filter_complex "[2:a]aformat=channel_layouts=stereo,aresample=48000[mus];[1:a]aformat=channel_layouts=stereo,aresample=48000[vo];[mus][vo]sidechaincompress=threshold=0.05:ratio=8:attack=15:release=300:makeup=2[mus_ducked];[vo][mus_ducked]amix=inputs=2:duration=first:weights=1 0.55:dropout_transition=0[mix];[mix]loudnorm=I={LUFS}:TP={TP}:LRA=11[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k -ar 48000 -movflags +faststart final.mp4`
   Map ONLY 0:v (ignore the VO-preview's baked audio) to avoid double VO.
3. (Optional) Run a measurement pass (`ffmpeg -i final.mp4 -af ebur128 -f null -`) and parse integrated loudness → `loudness_lufs`.
4. Upload to S3 key `phase4/{show_id}/{episode_id}/final/{TITLE_SAFE}_FINAL_v{final_master_version}.mp4`; presign (7-day).

STORE
- `state["final_master_s3_key"]`, `state["final_master_s3_url"]`, `state["loudness_lufs"]`, `state["current_agent"]="agent8"`.
- Append `{version: final_master_version, s3_key, s3_url, lufs}` to `final_assemblies.final_masters[]`; set `agent_outputs.agent8`.
- `production_pipelines`: `agent8_status:"completed"`, `current_agent:"agent_8"`. If delivery is disabled, also set `pipeline_status:"completed"` on both the job and the final_assemblies doc.

VERSIONING: final_master_version starts at 0; increment on re-run.

ROUTING: return state. The graph edge goes to agent_9 if delivery is enabled (config flag `enable_delivery`, default True), else to END. Expose `route_after_mix(state)->str` returning "deliver"|"end".

REQUIREMENTS: expose tunable sidechain/mix params (threshold, ratio, attack, release, music weight) as constants, stream ffmpeg stderr to logs, raise on non-zero exit, type hints, docstrings, temp cleanup + client close in finally. Deliver the complete file.
````

---

## 11. Agent 9 — Delivery & Captions *(NEW, optional)*

| Field | Value |
|---|---|
| Class | `DeliveryAgent` |
| File | `phase_4_agents/agent_9_delivery.py` |
| Model | None (deterministic from VO plan + FFmpeg) |
| LangGraph node | `agent_9_delivery_node` |
| Output | Captions (SRT) + platform exports + final metadata (S3 + MongoDB) |

**Role.** Finish for distribution. Generate **burned-in / sidecar captions** directly from Agent 5's `optimized_script` + `slice_guide` timestamps (no extra model needed — the script and timings are already exact). Optionally render **per-platform exports** (all 9:16 here, but you can add 1:1 / 16:9 variants) and write **final metadata** to MongoDB.

**Captions (SRT) from the slice guide:** each `Slice{start_sec, end_sec, text}` becomes one SRT cue. Optionally burn in with FFmpeg `subtitles=captions.srt:force_style='Fontsize=...,Outline=...'` for a hard-subbed export.

**Exports:** for each platform in `target_platforms`, copy/encode the final master to the platform's spec (default: all share the 9:16 master). Store each under the final prefix with a platform suffix.

**S3 keys:**
- Captions: `phase4/{show_id}/{episode_id}/final/{TITLE_SAFE}_v{final_master_version}.srt`
- Exports: `phase4/{show_id}/{episode_id}/final/{TITLE_SAFE}_FINAL_{platform}_v{final_master_version}.mp4`

**Storage:** `state["captions_s3_key"]`, `state["platform_exports"]`, `state["final_metadata"]`; set `final_assemblies.deliverables` and `pipeline_status:"completed"`; `agent_outputs.agent9`.
**Routing:** → `END`.

````text
You are building **Agent 9 — Delivery & Captions** for Phase 4 of a LangGraph video pipeline. Output one complete Python module.

GOAL
Finalize the ad for distribution: generate captions (SRT) deterministically from the VO plan's optimized script + slice-guide timestamps, optionally render per-platform exports, write final metadata, and upload everything to S3 + MongoDB.

FILE/CLASS/NODE
- File: `phase_4_agents/agent_9_delivery.py`
- Class: `DeliveryAgent`
- Node fn: `agent_9_delivery_node(state: Phase4State) -> Phase4State`

STACK: subprocess→ffmpeg, boto3, pymongo. No LLM. Fresh clients inside the node. Cleanup in finally.

INPUT (read from state): `final_master_s3_key`, `vo_plan` (has `slice_guide: [{start_sec,end_sec,text,aligns_with}]` and `optimized_script`), `target_platforms`, env aspect/resolution targets. Config flags: `burn_in_captions` (default False → produce sidecar SRT; True → also produce a hard-subbed export), `enable_exports` (default True).

PROCESSING
1. Build an SRT string from `vo_plan["slice_guide"]`: one cue per slice, index 1..N, `HH:MM:SS,mmm --> HH:MM:SS,mmm` from start_sec/end_sec, text = slice.text. (If slice_guide is empty, fall back to a single full-length cue with optimized_script.) Upload SRT to `phase4/{show_id}/{episode_id}/final/{TITLE_SAFE}_v{final_master_version}.srt`.
2. Download the final master (fresh presigned from final_master_s3_key).
3. If burn_in_captions: produce a hard-subbed copy `ffmpeg -i final.mp4 -vf "subtitles={srt}:force_style='Fontsize=42,Outline=3,Alignment=2,MarginV=120'" -c:a copy final_cc.mp4` (target platform export below uses this if enabled).
4. If enable_exports: for each platform in target_platforms, produce `{TITLE_SAFE}_FINAL_{platform}_v{ver}.mp4`. Default: all platforms are 9:16 and can be a stream-copy of the master (or the burned-in version). (Leave clear hooks to add 1:1/16:9 reframes later via the same FIT_FILTER approach as Agent 2.) Upload each; collect `{platform: presigned_url}`.
5. Assemble `final_metadata`: {title, episode_id, duration, lufs (from state), clip_count (len clip_manifest), final_master_s3_key, captions_s3_key, exports, created_at}.

STORE
- `state["captions_s3_key"]`, `state["platform_exports"]`, `state["final_metadata"]`, `state["current_agent"]="agent9"`, `state["pipeline_status"]="completed"`.
- Set `final_assemblies.deliverables = {captions_srt, exports}`; set `final_assemblies.pipeline_status="completed"`, `final_metadata`; set `agent_outputs.agent9`.
- `production_pipelines`: `agent9_status:"completed"`, `current_agent:"agent_9"`, `pipeline_status:"completed"`.

ROUTING: return state (edge → END).

REQUIREMENTS: correct SRT timecode formatting, type hints, docstrings, stream ffmpeg stderr to logs, temp cleanup + client close in finally. Deliver the complete file.
````

---

## 12. LangGraph Workflow Wiring (`langgraph_workflow.py`)

| Field | Value |
|---|---|
| File | `phase_4_agents/langgraph_workflow.py` |
| State | `Phase4State` |
| Entry | `initialize_node` |
| Invocation | `run_phase4_pipeline(show_id, episode_number, episode_id, *, movie_id=None, job_id=None, **opts)` |

**Graph edges:**

```
initialize_node            ─► agent_1_edl_node
agent_1_edl_node           ─► agent_2_assembly_node
agent_2_assembly_node      ─► agent_3_review_node
agent_3_review_node        ─► (conditional: route_after_review)
                               "revise" ─► agent_4_revise_node
                               "vo"     ─► agent_5_vo_node
agent_4_revise_node        ─► agent_2_assembly_node      (loop; cap enforced via edit_loop_count>=2 → force "vo")
agent_5_vo_node            ─► agent_6_av_merge_node
agent_6_av_merge_node      ─► agent_7_music_node
agent_7_music_node         ─► agent_8_final_mix_node
agent_8_final_mix_node     ─► (conditional: route_after_mix) "deliver" ─► agent_9_delivery_node | "end" ─► END
agent_9_delivery_node      ─► END
```

````text
You are building the **LangGraph workflow** that wires Phase 4 together. Output one complete Python module: `phase_4_agents/langgraph_workflow.py`.

CONTEXT
Phase 4 assembles approved per-shot videos into one finished DR ad. All node functions already exist in sibling modules and operate on `Phase4State`:
- initialize_node (this file)
- agent_1_edl_node                (agent_1_edl_generator)
- agent_2_assembly_node           (agent_2_assembly)
- agent_3_review_node + route_after_review  (agent_3_review)
- agent_4_revise_node             (agent_4_timestamp_reviser)
- agent_5_vo_node                 (agent_5_voiceover)
- agent_6_av_merge_node           (agent_6_av_merge)
- agent_7_music_node              (agent_7_music_score)
- agent_8_final_mix_node + route_after_mix  (agent_8_final_mix)
- agent_9_delivery_node           (agent_9_delivery)

STACK: LangGraph StateGraph/CompiledStateGraph; Celery fork model (nodes open their own DB/S3 clients). Pydantic v2 elsewhere.

BUILD
1. Implement `initialize_node` per the initialize spec I provide (gather approved clips → clip_manifest, load script/shotlist/title, set defaults, upsert final_assemblies, set job running). [Paste the initialize build prompt's steps here, or import if already built.]
2. Construct the StateGraph(Phase4State), add all nodes, set entry to initialize_node, and add the edges exactly as in the graph above. Use `add_conditional_edges("agent_3_review_node", route_after_review, {"revise":"agent_4_revise_node","vo":"agent_5_vo_node"})` and `add_conditional_edges("agent_8_final_mix_node", route_after_mix, {"deliver":"agent_9_delivery_node","end": END})`. agent_4 → agent_2 (loop). The 2-loop cap lives inside route_after_review (force "vo" when edit_loop_count>=2).
3. Compile the graph. Expose `build_phase4_graph() -> CompiledStateGraph`.
4. Expose `run_phase4_pipeline(show_id, episode_number, episode_id, *, movie_id=None, project_id=None, job_id=None, aspect_ratio=None, target_platforms=None, enable_delivery=True, keep_clip_audio=False, burn_in_captions=False) -> Phase4State` that seeds an initial Phase4State dict and invokes the compiled graph. Return the final state.
5. Wrap invocation in try/except: on exception set pipeline_status="failed", record into final_assemblies + production_pipelines, re-raise.

REQUIREMENTS: type hints, docstrings, structured logging at each node boundary, and a `__main__` smoke test that prints the resolved graph nodes/edges. Deliver the complete file.
````

---

## 13. Cross-Cutting Infrastructure

### S3 Key Patterns

Bucket `zeroshot-v1`, region `eu-north-1`.

**Phase 3 INPUT (read-only; key is parsed from the stored presigned URL, never reconstructed):**

| Content | Pattern |
|---|---|
| Phase 3 per-shot video | `phase3/{project_folder}/generated_videos/scene_{N}_shot_{M}_v{V}.mp4` (filename version starts at **v1**) |

**Phase 4 OUTPUT:**

| Content | Pattern |
|---|---|
| Rough cut (initial + loops) | `phase4/{show_id}/{episode_id}/rough_cut/v{n}.mp4` |
| (Optional) trimmed segments | `phase4/{show_id}/{episode_id}/segments/{shot_id}_{version}_{order}.mp4` |
| Voiceover audio | `phase4/{show_id}/{episode_id}/audio/{TITLE_SAFE}_VO_v{n}.wav` |
| Music / score audio | `phase4/{show_id}/{episode_id}/audio/{TITLE_SAFE}_MUSIC_v{n}.wav` |
| VO preview (video + VO) | `phase4/{show_id}/{episode_id}/with_vo/v{n}.mp4` |
| Final master | `phase4/{show_id}/{episode_id}/final/{TITLE_SAFE}_FINAL_v{n}.mp4` |
| Platform export | `phase4/{show_id}/{episode_id}/final/{TITLE_SAFE}_FINAL_{platform}_v{n}.mp4` |
| Captions | `phase4/{show_id}/{episode_id}/final/{TITLE_SAFE}_v{n}.srt` |

All S3 objects use **presigned URLs, 7-day expiry** (`ExpiresIn=86400 * 7`). **Always regenerate presigned URLs from the stored `s3_key`** before download — never reuse stored URLs from earlier phases.

### MongoDB Write Patterns

**`final_assemblies`** (per-agent output, mirrors the `agent_outputs.agent{N}` convention):

```
{
  "agent_outputs.agent{N}.status": "completed",
  "agent_outputs.agent{N}.executed_at": ISODate(),
  "agent_outputs.agent{N}.output": { /* agent data */ },
  "updated_at": ISODate()
}
```

Versioned arrays appended per pass: `edl_versions[]`, `rough_cuts[]`, `reviews[]`, `vo[]`, `vo_preview[]`, `music[]`, `final_masters[]`; final `deliverables` + `final_metadata` set by Agent 9.

**`production_pipelines`** (job tracker, reused):

```
{
  "agent{N}_status": "running" | "completed" | "failed" | "skipped" | "retrying",
  "current_agent": "agent_{N}",
  "pipeline_status": "running" | "completed" | "failed",
  "updated_at": ISODate()
}
```

### Loop Caps (Phase 4)

| Loop | Cap | State key |
|---|---|---|
| Edit-review loop (Agent 4 → Agent 2 → Agent 3) | **2** | `edit_loop_count` (force-pass to VO at limit) |

Other agents are single-pass within a run; re-running the pipeline bumps the relevant version counters.

---

## 14. Naming Conventions & Nomenclature (Phase 4 Additions)

**Title-safe names** (for S3 filenames; reuses the Phase 1 `safe_name` convention):
```
TITLE_SAFE = title.upper().replace(" ", "_").replace("/", "_")
# "Glow Serum Launch" → "GLOW_SERUM_LAUNCH"
```

**Shot & clip identity.** A shot is `scene_{N}_shot_{M}` (e.g. `scene_2_shot_1`). A clip reference is the triple **`{s3_key, version, filename}`** in `ClipRef`, where `s3_key` is **parsed from the stored presigned URL** in `shots.video.v{N}.generated_videos_s3[]` and `version` is the **filename** `_v{V}` (starts at v1) — distinct from the MongoDB attempt key `attempt_key` (`video.v0`/`v1`). Downstream agents fetch **by exact `s3_key`**; descriptions are LLM context only. This is the mechanism that distinguishes multiple versions of the same shot in S3.

**Phase 4 version keys:**

| Key | Meaning |
|---|---|
| `rough_cut v0` | First assembly (Agent 2 from Agent 1's EDL) |
| `rough_cut v1, v2` | Re-assemblies after Agent 4 revision loops |
| `VO v0`, `MUSIC v0` | First VO / music generation (bump on re-runs) |
| `with_vo v0` | First VO preview (Agent 6) |
| `FINAL v0` | First final master (Agent 8); `_{platform}` suffix for exports |

> Phase 4 **output** versions above are 0-based (internal to Phase 4). Phase 3 **input** clip versions are 1-based in the filename (`_v1`, `_v2`, …) and are read, not generated, here.

**Pipeline status values (Phase 4):** `pending → running → completed | failed`. The one human checkpoint is **Agent 0** (Phase 3 → Phase 4 gate, external Streamlit tool — not a graph node); the Phase 4 graph itself runs straight through from `initialize_node`. A future optional in-graph human-approval node (e.g. before Agent 5) could reuse the same resume pattern.

**Review decision values:**

| Stage | Values |
|---|---|
| Agent 3 (Final-Cut Review) | `approved`, `edit` |

**Model strings (Phase 4):** `gemini-3.1-pro-preview` (EDL/review/revision/directing, video understanding), `gemini-3.1-flash-tts-preview` (VO), `lyria-3` (music).

---

### Appendix — Build Order

Recommended order to feed the build prompts to Claude (each is self-contained):
`workflow_state.py` (paste the `Phase4State` + `ClipRef` + `ShotCandidates` blocks) → **Agent 0 backend endpoints** (`/api/v1/phase4/...`) → **`tools/video_review.py`** (the checkpoint UI; also delivered as a standalone file) → Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → Agent 6 → Agent 7 → Agent 8 → Agent 9 → `langgraph_workflow.py` (initialize + wiring). Build the `final_assemblies_service` (mirroring your existing `assets_collection_service`/`project_service` `update_agent_output` helpers) before wiring, or stub it and fill in.

Agent 0 runs **before** the graph: humans review in `video_review.py`, selections land in `shots.<shot>.video_review_selection`, and `POST /master/continue-to-phase4/{job}` starts the Phase 4 graph (whose `initialize_node` reads those selections).
