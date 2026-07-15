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

    # ---- Agent 5A: VO Director ----
    vo_director_plan: Dict          # VODirectorPlan — persona, scene, director_notes,
                                    # sample_context, transcript_with_tags, visual_breakdown,
                                    # word_count_math, slice_guide, recommended_voice

    # ---- Agent 5B: TTS ----
    vo_tts_prompt: str              # Assembled advanced prompt string (for debugging/reruns)
    vo_s3_url: str
    vo_s3_key: str
    vo_version: int
    vo_duration: float

    # ---- Agent 6: A/V merge (VO preview) ----
    vo_preview_s3_url: str
    vo_preview_s3_key: str
    vo_preview_version: int

    # ---- Agent 7A: Music Director ----
    music_director_plan: Dict       # MusicDirectorPlan — filled_prompt, duration_sec,
                                    # timing_map, volume_envelope

    # ---- Agent 7B: Lyria Generator ----
    music_prompt: str               # filled Lyria prompt (mirrored from music_director_plan)
    music_plan: List[Dict]          # timing_map (mirrored for downstream compat)
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


class ShotCandidates(TypedDict):
    shot_id: str                 # "scene_1_shot_1"
    scene_number: int
    shot_number: int
    selection_mode: str          # "single"     → use candidates[0] as-is (human picked exactly one)
                                 # "choose_one" → Agent 1 MUST pick exactly one of candidates
                                 #   (human picked 2+, OR human picked none → fallback: all versions)
    selection_source: str        # "human" | "fallback_all_versions"
    candidates: List[ClipRef]    # 1 if "single"; >=1 if "choose_one"
