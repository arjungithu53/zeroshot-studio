# Production Pipeline — Product Documentation

> **Scope:** `backend/services/production/app/services/`
> This document covers every agent, phase, input/output schema, storage pattern, model, and naming convention in the production AI pipeline.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Phase 1: Asset Generation Pipeline](#2-phase-1-asset-generation-pipeline)
   - [Agent 1 — Asset Generator](#agent-1--asset-generator)
   - [Agent 2 — Asset Reviewer](#agent-2--asset-reviewer)
   - [Agent 3 — Prompt Generator](#agent-3--prompt-generator)
   - [Agent 4 — Prompt Optimizer](#agent-4--prompt-optimizer)
   - [Agent 5 — Image Generator](#agent-5--image-generator)
   - [Agent 6 — Image Reviewer](#agent-6--image-reviewer)
   - [Agent 7 — Image Editor](#agent-7--image-editor)
   - [Agent 8 — Variation Generator](#agent-8--variation-generator)
3. [Phase 2: Shot Image Pipeline](#3-phase-2-shot-image-pipeline)
   - [Agent 1 (P2) — Shot Strategy](#agent-1-p2--shot-strategy)
   - [Agent 2 (P2) — Image Prompt Generator](#agent-2-p2--image-prompt-generator)
   - [Agent 3 (P2) — Prompt Review](#agent-3-p2--prompt-review)
   - [Agent 12 — Shot Design](#agent-12--shot-design)
   - [Agent 13 — Prompt Modifier](#agent-13--prompt-modifier)
   - [Agent 14 — Imagen Generator (Initial)](#agent-14--imagen-generator-initial)
   - [Agent 15 — Image Reviewer (Phase 2)](#agent-15--image-reviewer-phase-2)
   - [Agent 15A — Prompt Regeneration](#agent-15a--prompt-regeneration)
   - [Agent 14 Regen — Imagen Generator (Regeneration)](#agent-14-regen--imagen-generator-regeneration)
   - [Agent 7 (P2) — Shot Editor](#agent-7-p2--shot-editor)
   - [Agents 16/17/18 — Product Fidelity Loop](#agents-161718--product-fidelity-loop)
4. [Phase 3: Video Generation Pipeline](#4-phase-3-video-generation-pipeline)
   - [Initialize Node](#initialize-node)
   - [Prompt Router Node](#prompt-router-node)
   - [Video Generation Node](#video-generation-node)
   - [AI Review Node](#ai-review-node)
   - [Human Checkpoint Node](#human-checkpoint-node)
5. [Cross-Cutting Infrastructure](#5-cross-cutting-infrastructure)
6. [Naming Conventions & Nomenclature](#6-naming-conventions--nomenclature)

---

## 1. System Overview

### Purpose

The production pipeline converts a script and shot list into a complete set of video ad assets: character/location/prop images with angle variations (Phase 1), per-shot composite images (Phase 2), and per-shot generated videos (Phase 3).

### High-Level Flow

```
Script
  └─► Phase 1 (8 agents) ─► Asset Images + Angle Variations (S3)
                                           │
Shot List ─────────────────────────────────►
  └─► Phase 2 (~12 agents) ─► Per-Shot Composite Images (S3)
                                           │
                              ─────────────►
                              Phase 3 (4 nodes/shot) ─► Videos (S3)
```

### Tech Stack

| Component | Technology |
|---|---|
| Workflow orchestration | LangGraph (`StateGraph`, `CompiledStateGraph`) |
| LLM for text | `gemini-3.1-pro-preview` via `google.genai` + `langchain_google_genai` |
| Image generation/editing | `gemini-3.1-flash-image-preview` ("Nano Banana") |
| Image generation (Phase 2) | Gemini Imagen 4.0 (via `google.genai`) |
| Video generation | Google Veo 3.1 (via `VideoGenerationAPIAgent`) |
| Image storage | AWS S3 (presigned URLs, 7-day expiry) |
| Metadata storage | MongoDB Atlas |
| Job queue | Celery (fork model — fresh DB connections per node) |
| Image processing | Pillow (PIL) |
| Data validation | Pydantic v2 |
| HTTP client | `requests` |

### MongoDB Collections

| Collection | Purpose |
|---|---|
| `production_projects` | Phase 1 projects; nested `agent_outputs.agent{N}` per agent |
| `assets_collections` | Movie-mode Phase 1 asset bundles; `agent{N}_output` per agent |
| `shots` | Phase 2+3 shot documents; versioned `image.v{N}` and `video.v{N}` inside `annotated_shots[]` |
| `production_pipelines` | Lightweight job trackers; `agent{N}_status`, `current_agent`, `pipeline_status` |
| `movies` | Movie-level settings; `global_settings.visual_style` consumed by Agent 13 |

### Environment Variables

```
GEMINI_API_KEY / GOOGLE_API_KEY        # Gemini, Imagen, Veo API access
production_AWS_ACCESS_KEY_ID           # S3 access key
production_AWS_SECRET_ACCESS_KEY       # S3 secret
production_S3_BUCKET_NAME              # S3 bucket name
production_AWS_REGION                  # Default: us-east-1
production_S3_ENDPOINT_URL             # Optional: S3-compatible endpoint
production_MONGODB_URI                 # Local MongoDB connection string
MONGODB_ATLAS_URI                      # Atlas connection string
production_ALLOW_LOCAL_MONGO           # "true" to prefer local URI
```

---

## 2. Phase 1: Asset Generation Pipeline

### Workflow Overview

| File | `phase_1_agents/langgraph_workflow.py` |
|---|---|
| State type | `Phase1State` (TypedDict, `workflow_state.py`) |
| Entry point | `agent_1_node` |
| Orchestration | LangGraph `StateGraph` |

#### Phase 1 Agent Flow

```
agent_1_node
    └─► agent_2_node
            └─► agent_3_node
                    └─► agent_4_node
                            └─► agent_5_node
                                    └─► agent_6_node
                                            ├─► regeneration_router_node ─► agent_5_node (loop, max 3)
                                            ├─► agent_7_node ─► agent_6_node (re-review, max 3 loops)
                                            └─► agent_8_node (all approved)
```

#### Phase 1 State Schema (`Phase1State`)

```python
class Phase1State(TypedDict):
    # Inputs
    script_path: str
    script_content: str
    project_id: Optional[str]          # Legacy mode: save to production_projects
    movie_id: Optional[str]            # Movie workflow mode
    assets_collection_id: Optional[str]  # Movie workflow mode
    job_id: Optional[str]              # Pipeline job ID for status tracking
    visual_style: Optional[str]        # "realistic" | "pixar" | "2d"
    csv_entity_mapping: Optional[Dict] # Pre-extracted entities from shot list CSV
    product_prop: Optional[Dict]       # Pre-built product prop (legacy path)
    product_image_s3_url: Optional[str]  # Uploaded product image S3 URL

    # Per-agent outputs (each written by that agent's node)
    extracted_assets: Dict[str, List[Dict]]    # Agent 1
    enhanced_assets: Dict[str, List[Dict]]     # Agent 2
    generated_prompts: Dict[str, List[Dict]]   # Agent 3
    optimized_prompts: Dict[str, List[Dict]]   # Agent 4
    generated_images: Dict[str, List[Dict]]    # Agent 5
    image_reviews: Dict[str, List[Dict]]       # Agent 6
    edited_images: Dict[str, List[Dict]]       # Agent 7
    variation_images: Dict[str, List[Dict]]    # Agent 8

    # Control
    current_agent: str
    pipeline_status: str
    needs_editing_assets: List[str]        # Format: "characters:uuid"
    needs_regeneration_assets: List[str]   # Format: "characters:uuid"
    approved_asset_ids: Optional[List[str]]
    auto_edit_count: int                   # Max 3
    auto_regeneration_count: int           # Max 3
    # ...additional control fields
```

#### Storage Pattern (Phase 1)

Every agent node calls one of two save helpers after completing:

```python
# Movie workflow (assets_collection_id present)
assets_collection_service.update_agent_output(
    assets_collection_id=state["assets_collection_id"],
    agent_number=N,
    status="completed",
    output={...}
)

# Legacy workflow (project_id present)
project_service.update_agent_output(
    project_id=state["project_id"],
    agent_number=N,
    status="completed",
    output={...}
)
```

MongoDB path (legacy): `production_projects.agent_outputs.agent{N}.{status, output, executed_at}`

Job status updates are sent via `PipelineService.update_job_status(job_id, ...)` to `production_pipelines`.

---

### Agent 1 — Asset Generator

| Field | Value |
|---|---|
| Class | `AssetGeneratorAgent` |
| File | `phase_1_agents/agent_1_asset_generator.py` |
| Model | `gemini-3.1-pro-preview` |
| Output format | Structured JSON via Pydantic `ExtractedAssets` |
| LangGraph node | `agent_1_node` |

#### Input

Fetched from `Phase1State`:

| Key | Type | Source |
|---|---|---|
| `script_content` | `str` | Script text passed into pipeline |
| `csv_entity_mapping` | `Optional[Dict]` | Pre-processed from shot list CSV; keys: `unique_characters`, `unique_locations`, `has_entity_data`, `character_shots`, `location_shots`, `product_shot_numbers` |
| `product_image_s3_url` | `Optional[str]` | Uploaded product image; triggers `product_image_available=True` flag |

#### LLM Call

```python
response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents=Agent1Prompts.asset_extraction(
        script_content, csv_entity_mapping, product_image_available
    ),
    config={
        "response_mime_type": "application/json",
        "response_schema": ExtractedAssets,
    }
)
extracted: ExtractedAssets = response.parsed
```

#### Output Schema

```
ExtractedAssets
├── characters: List[Character]
│     Character:
│       id: str (UUID v4, auto-generated)
│       name: str
│       description: str
│       age_range: Optional[str]
│       gender: Optional[str]
│       key_features: List[str]
│       clothing_style: Optional[str]
│       role: str  — "protagonist" | "antagonist" | "supporting"
│       scenes: List[str]
│       importance: str  — "critical" | "important" | "background"
│       csv_name: str  (added post-parse when CSV mapping present)
│
├── locations: List[Location]
│     Location:
│       id: str (UUID v4)
│       name: str
│       description: str
│       setting_type: str  — "interior" | "exterior"
│       time_of_day: Optional[str]
│       weather: Optional[str]
│       lighting: Optional[str]
│       atmosphere: Optional[str]
│       key_visual_elements: List[str]
│       scenes: List[str]
│       importance: str
│       csv_name: str  (added post-parse when CSV mapping present)
│
└── props: List[Prop]
      Prop:
        id: str (UUID v4)
        name: str
        description: str
        material: Optional[str]
        size: Optional[str]  — "small" | "medium" | "large"
        condition: Optional[str]
        usage: str
        scenes: List[str]
        importance: str
        is_product: bool  (set when product_image_available=True and prop flagged)
        pre_generated_image_url: str  (set to product_image_s3_url if is_product)
```

#### Post-Processing in LangGraph Node

1. `csv_name` field added to characters/locations matching CSV names
2. If `product_image_s3_url` present: attach to prop flagged `is_product=True` by agent, OR inject fallback `PRODUCT` prop if agent didn't flag one
3. CSV validation: `agent.validate_csv_mapping()` checks for missing/extra entities
4. Human feedback applied (or auto-approved)

#### Storage

- **State key**: `Phase1State["extracted_assets"]`
- **MongoDB**: `update_agent_output(agent_number=1, status="completed", output={"extracted_assets": {...}})`

---

### Agent 2 — Asset Reviewer

| Field | Value |
|---|---|
| Class | `AssetReviewerAgent` |
| File | `phase_1_agents/agent_2_asset_reviewer.py` |
| Model | `gemini-3.1-pro-preview` |
| Output format | Structured JSON via Pydantic `AssetReviewReport` |
| LangGraph node | `agent_2_node` |

#### Input

| Key | Type | Source |
|---|---|---|
| `extracted_assets` | `Dict[str, List[Dict]]` | `Phase1State["extracted_assets"]` from Agent 1 |
| `script_content` | `str` | `Phase1State["script_content"]` |

#### LLM Call

```python
response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents=Agent2Prompts.asset_review(script_content, original_assets),
    config={
        "response_mime_type": "application/json",
        "response_schema": AssetReviewReport,
    }
)
review: AssetReviewReport = response.parsed
```

#### Output Schema

```
AssetReviewReport
├── completeness_check: CompletenessCheck
│     missing_characters: List[MissingAsset{id, name, reason, description, importance}]
│     missing_locations: List[MissingAsset{...}]
│     missing_props: List[MissingAsset{...}]
│
├── duplicates_detected: DuplicatesDetected
│     characters: List[DuplicateAsset{duplicate_names, reason, merged_name, merged_description}]
│     locations: List[DuplicateAsset{...}]
│     props: List[DuplicateAsset{...}]
│
├── description_enhancements: DescriptionEnhancements
│     characters: List[DescriptionEnhancement{asset_name, original_description, enhanced_description, improvements_made}]
│     locations: List[DescriptionEnhancement{...}]
│     props: List[DescriptionEnhancement{...}]
│
├── accuracy_issues: List[AccuracyIssue{asset_type, asset_name, issue, suggested_fix}]
├── edge_cases: List[EdgeCase{asset_name, case_type, description, recommendation}]
├── overall_quality_score: QualityScores{completeness, accuracy, detail_level, production_readiness}  (all 0–100)
└── recommendations: List[str]
```

#### Post-Processing in LangGraph Node

Auto-approve: all enhancements applied, all missing assets added. **CSV guard**: characters/locations added by Agent 2 that are NOT in the CSV `unique_characters`/`unique_locations` sets are silently removed — Agent 2 has no CSV context and can hallucinate phantom entities.

`enhanced_assets` retains the same structure as `extracted_assets` but with enriched `description` fields.

#### Storage

- **State key**: `Phase1State["enhanced_assets"]`
- **MongoDB**: `update_agent_output(agent_number=2, output={"review_results": {...}, "enhanced_assets": {...}})`

---

### Agent 3 — Prompt Generator

| Field | Value |
|---|---|
| Class | `PromptGeneratorAgent` |
| File | `phase_1_agents/agent_3_prompt_generator.py` |
| Model | `gemini-3.1-pro-preview` |
| Output format | Structured JSON per asset (Character/Location/Prop) |
| LangGraph node | `agent_3_node` |

#### Input

| Key | Type | Source |
|---|---|---|
| `enhanced_assets` | `Dict[str, List[Dict]]` | `Phase1State["enhanced_assets"]` from Agent 2 |
| `visual_style` | `str` | `Phase1State["visual_style"]` — `"realistic"` \| `"pixar"` \| `"2d"` |

#### Visual Style Preamble Injected per Prompt

| Style | Required prefix | Forbidden keywords |
|---|---|---|
| `realistic` | `"Raw, unretouched photograph"` or `"Candid documentary photo"` | `photorealistic`, `hyperrealistic`, `8K`, `masterpiece`, `cinematic`, `3D`, `render` |
| `pixar` | `"Pixar-style 3D animation"` or `"Disney Pixar style"` | `realistic`, `photorealistic`, `photography`, `DSLR` |
| `2d` | `"2D animation style"` or `"hand-drawn animation"` | `realistic`, `photorealistic`, `3D`, `CGI` |

#### LLM Calls (one per asset)

```python
# For characters
response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents=character_prompt_request,
    config={"response_mime_type": "application/json", "response_schema": CharacterPromptData}
)

# For locations
response = client.models.generate_content(..., config={..., "response_schema": LocationPromptData})

# For props
response = client.models.generate_content(..., config={..., "response_schema": PropPromptData})
```

#### Output Schema (per asset)

```
CharacterPromptData / LocationPromptData / PropPromptData
├── character_name / location_name / prop_name: str
└── master_prompt: MasterPrompt
      ├── initial_prompt: str  (150–300 words for chars/locs; 100–200 for props)
      ├── negative_prompt: str
      ├── technical_specs: TechnicalSpecs
      │     ├── aspect_ratio: str  e.g. "3:4", "16:9", "1:1"
      │     ├── camera_angle: str
      │     ├── framing: str
      │     ├── lighting: str
      │     └── style_keywords: List[str]
      └── recommended_settings: RecommendedSettings
            ├── model: str
            ├── steps: str
            └── guidance_scale: str
```

After Pydantic parse, `id` (UUID from Agent 1) and `name` are added to each dict.

#### State Key Output

```python
Phase1State["generated_prompts"] = {
    "characters": [{"id": "uuid", "name": "NAME", "master_prompt": {...}, ...}],
    "locations": [...],
    "props": [...]
}
```

#### Storage

- **State key**: `Phase1State["generated_prompts"]`
- **MongoDB**: `update_agent_output(agent_number=3, output={"generated_prompts": {...}})`

---

### Agent 4 — Prompt Optimizer

| Field | Value |
|---|---|
| Class | `PromptOptimizerAgent` |
| File | `phase_1_agents/agent_4_prompt_optimizer.py` |
| Model | `gemini-3.1-pro-preview` |
| Output format | Structured JSON via Pydantic `OptimizedPromptData` |
| LangGraph node | `agent_4_node` |

#### Input

| Key | Type | Source |
|---|---|---|
| `generated_prompts` | `Dict[str, List[Dict]]` | `Phase1State["generated_prompts"]` from Agent 3 |
| `visual_style` | `str` | `Phase1State["visual_style"]` |

#### LLM Call (one per asset)

```python
response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents=Agent4Prompts.prompt_optimization(asset_name, asset_type, initial_prompt_data),
    config={"response_mime_type": "application/json", "response_schema": OptimizedPromptData}
)
```

#### Output Schema

```
OptimizedPromptData
├── asset_name: str
├── asset_type: str  — "character" | "location" | "prop"
├── optimization_analysis: OptimizationAnalysis
│     ├── strengths: List[str]
│     ├── improvements_needed: List[str]
│     └── added_elements: List[str]
├── final_prompt: FinalPrompt
│     ├── prompt: str  (200–350 words)
│     ├── negative_prompt: str
│     ├── technical_specs: TechnicalSpecs  (same as Agent 3)
│     └── recommended_settings: RecommendedSettings
└── comparison: PromptComparison
      ├── initial_word_count: str
      ├── final_word_count: str
      ├── detail_level_improvement: str
      └── key_changes: List[str]
```

`id` and `name` from Agent 3 are preserved in the dict.

#### State Key Output

```python
Phase1State["optimized_prompts"] = {
    "characters": [{"id": "uuid", "name": "NAME", "final_prompt": {"prompt": "...", "negative_prompt": "...", "technical_specs": {...}}, ...}],
    "locations": [...],
    "props": [...]
}
```

#### Storage

- **State key**: `Phase1State["optimized_prompts"]`
- **MongoDB**: `update_agent_output(agent_number=4, output={"optimized_prompts": {...}})`

---

### Agent 5 — Image Generator

| Field | Value |
|---|---|
| Class | `ImageGeneratorAgent` |
| File | `phase_1_agents/agent_5_image_generator.py` |
| Model | `gemini-3.1-flash-image-preview` ("Nano Banana") |
| LangGraph node | `agent_5_node` |
| Output storage | AWS S3 only (no local files for main path) |

#### Input

| Key | Type | Source |
|---|---|---|
| `optimized_prompts` | `Dict[str, List[Dict]]` | `Phase1State["optimized_prompts"]` from Agent 4 |
| `assets_to_regenerate` | `Optional[List[str]]` | Set during regeneration loops; format: `"characters:uuid"` |

#### Image Generation Call

```python
response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=full_prompt,   # prompt + "\n\nDO NOT INCLUDE: " + negative_prompt
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio)  # "1:1"|"16:9"|"9:16"|"3:4"|"4:3"
    )
)
# Extract image from response.candidates[0].content.parts[].inline_data.data
```

#### S3 Upload

```python
# Key pattern
s3_key = f"phase1/generated_images/{timestamp}_{filename}"
# e.g. phase1/generated_images/20251006_160946_BLACK_LAB_PUPPY_1.png

s3_client.put_object(Bucket=bucket, Key=s3_key, Body=image_bytes, ContentType="image/png")
s3_url = s3_client.generate_presigned_url("get_object",
    Params={"Bucket": bucket, "Key": s3_key},
    ExpiresIn=86400 * 7  # 7 days
)
```

#### Special Cases

- **Product props** (`is_product=True`): Excluded from generation. Uploaded S3 URL injected directly as:
  ```python
  {"index": 1, "url": product_s3_url, "s3_url": product_s3_url, "source": "uploaded_product_image"}
  ```
- **Regeneration mode**: Only assets in `assets_to_regenerate` are (re)generated. Previous images for non-regenerated assets are preserved via `_merge_generated_images_with_versioning()`, which archives old images under a `versions` array.

#### Output Schema

```python
Phase1State["generated_images"] = {
    "characters": [{
        "id": "uuid",
        "name": "ASSET NAME",
        "prompt": "full prompt text",
        "negative_prompt": "...",
        "aspect_ratio": "1:1",
        "technical_specs": {...},
        "images": [{
            "index": 1,
            "url": "https://s3...presigned...",
            "s3_url": "https://s3...presigned...",
            "s3_key": "phase1/generated_images/...",
            "filename": "SAFE_NAME_1.png"
        }],
        "generation_timestamp": "2025-10-06T16:09:46.123456",
        # Regeneration versioning (added on regen):
        "versions": [{"version": 0, "images": [...], "archived_at": "..."}],
        "current_version": 1,
        "regenerated_at": "..."
    }],
    "locations": [...],
    "props": [...]
}
```

**Failed generations** stored in `Phase1State["failed_generations"]`:
```python
[{"asset_id": "uuid", "asset_name": "NAME", "asset_type": "character", "reason": "...", "prompt": "...", ...}]
```

#### Storage

- **State key**: `Phase1State["generated_images"]`
- **MongoDB**: `update_agent_output(agent_number=5, output={"generated_images": {...}, "failed_generations": [...]})`

---

### Agent 6 — Image Reviewer

| Field | Value |
|---|---|
| Class | `ImageReviewerAgent` |
| File | `phase_1_agents/agent_6_image_reviewer.py` |
| Model | `gemini-3.1-pro-preview` (multimodal: vision) |
| Output format | Structured JSON via Pydantic `ImageReviewResult` |
| LangGraph node | `agent_6_node` |

#### Input

| Key | Type | Source |
|---|---|---|
| `generated_images` | `Dict[str, List[Dict]]` | `Phase1State["generated_images"]` from Agent 5 |
| `optimized_prompts` | `Dict[str, List[Dict]]` | `Phase1State["optimized_prompts"]` from Agent 4 |
| `recently_edited_asset_ids` | `Optional[List[str]]` | Set by Agent 7 for selective re-review |
| `image_reviews` | `Optional[Dict]` | Previous reviews (preserved for assets not being re-reviewed) |

#### Image Loading

```python
# S3 URL: fetch via requests
image_data = requests.get(s3_url).content

# Multi-part Gemini call
content_parts = [
    types.Part.from_text(text=review_prompt),
    types.Part.from_bytes(data=image_data, mime_type="image/png")
]
response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents=content_parts,
    config={"response_mime_type": "application/json", "response_schema": ImageReviewResult}
)
```

#### Scoring Rubric

| Criterion | Max Points | Notes |
|---|---|---|
| Prompt accuracy | 40 | How well image matches prompt |
| Background compliance | 30 | Characters/props: neutral bg required; graduated 0–30; locations: full environment expected |
| Technical quality | 20 | Sharpness, lighting, no major artifacts |
| Production readiness | 10 | Suitable for I2V compositing |
| **Total** | **100** | 70+ = approved, 50–69 = needs_edit, 0–49 = regenerate |

**Product prop override**: If `is_product=True` and decision = `"regenerate"`, overridden to `"approved"` (fidelity lock).

#### Output Schema

```
ImageReviewResult (per image)
├── asset_name: str
├── asset_type: str
├── image_index: int
├── decision: str  — "approved" | "needs_edit" | "regenerate"
├── overall_score: int  (0–100)
├── scores: ImageReviewScores
│     ├── prompt_accuracy: int  (0–40)
│     ├── background_compliance: int  (0–30)
│     ├── technical_quality: int  (0–20)
│     └── production_readiness: int  (0–10)
├── assessment: ImageAssessment
│     ├── strengths: List[str]
│     ├── issues: List[str]
│     ├── missing_elements: List[str]
│     └── ai_artifacts: List[str]
├── feedback: ImageReviewFeedback
│     ├── for_edit: str
│     ├── for_regeneration: str
│     └── general_notes: str
└── production_notes: ProductionNotes
      ├── compositing_ready: bool
      ├── concerns: List[str]
      └── recommendations: List[str]
```

#### State Key Output

```python
Phase1State["image_reviews"] = {
    "characters": [{"id": "uuid", "name": "NAME", "reviews": [ImageReviewResult dict, ...]}],
    "locations": [...],
    "props": [...]
}

Phase1State["needs_editing_assets"] = ["characters:uuid1", "props:uuid2"]
Phase1State["needs_regeneration_assets"] = ["locations:uuid3"]
```

#### Routing Logic

- `needs_regeneration` → `regeneration_router_node` → back to `agent_5_node` (max 3 auto loops)
- `needs_editing` → `agent_7_node`
- All approved → `agent_8_node`

#### Prompt Rewriting (for regeneration)

```python
# Calls Gemini text model to rewrite the failing prompt
modified_prompts = agent.rewrite_prompts_for_regeneration(optimized_prompts)
# Stored in Phase1State["regenerated_prompts"]
```

#### Storage

- **State keys**: `Phase1State["image_reviews"]`, `needs_editing_assets`, `needs_regeneration_assets`
- **MongoDB**: `update_agent_output(agent_number=6, output={image_reviews, needs_editing_assets, needs_regeneration_assets})`

---

### Agent 7 — Image Editor

| Field | Value |
|---|---|
| Class | `ImageEditAgent` |
| File | `phase_1_agents/agent_7_image_editor.py` |
| Models | `gemini-3.1-pro-preview` (edit prompt gen) + `gemini-3.1-flash-image-preview` (image editing) |
| LangGraph node | `agent_7_node` |

#### Input

| Key | Type | Source |
|---|---|---|
| `image_reviews` | `Dict` | Agent 6 output (filtered to `needs_edit` decisions) |
| `generated_images` | `Dict` | Agent 5 output (provides image paths/URLs) |
| `optimized_prompts` | `Dict` | Agent 4 output (provides original prompt for context) |
| `needs_editing_assets` | `List[str]` | Format: `"characters:uuid"` |

#### Step 1 — Edit Prompt Generation

```python
# Uses gemini-3.1-pro-preview (text only) to produce targeted Nano Banana instruction
response = model.generate_content(edit_prompt_generation_request)
# Returns: {asset_name, asset_type, edit_prompt, edit_rationale, expected_changes, guidance_scale, edit_strength}
```

Edit prompt format guidelines:
- **Action verb** + **target region** + **desired result** + **explicit PRESERVE clause**
- Example: `"Replace the background behind the character with a smooth, solid neutral grey studio backdrop; preserve the character's exact pose, costume, face, and lighting completely unchanged"`
- If edit prompt contains `"NO EDIT RECOMMENDED"` → original image used as-is

**Product fidelity lock** prepended to all product prop edit prompts:
```
CRITICAL — PRODUCT FIDELITY: Under NO circumstances alter the product's shape, size, proportions,
text, logo, label, color, or branding. Permitted adjustments ONLY: background, shadows, reflections,
or lighting. The product itself must remain pixel-perfect.
```

#### Step 2 — Image Editing (Nano Banana)

```python
response = gemini_client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[edit_prompt, reference_PIL_image],
    config=types.GenerateContentConfig(response_modalities=["IMAGE"])
)
# Saves to: output/edited_images/{timestamp}/{asset_type}/{SAFE_NAME}_edited.png
```

#### Output Schema

```python
Phase1State["edited_images"] = {
    "characters:uuid:image_1": {
        "asset_id": "uuid",
        "asset_name": "NAME",
        "asset_type": "characters",
        "original_image": "path_or_url",
        "edit_prompt": "Replace the background...",
        "edit_prompt_data": {
            "asset_name": "NAME",
            "asset_type": "character",
            "edit_prompt": "...",
            "edit_rationale": "...",
            "expected_changes": ["..."],
            "guidance_scale": 2.5,
            "edit_strength": "moderate"
        },
        "edited_images": [{"index": 1, "local_path": "output/edited_images/...", "filename": "NAME_edited.png"}],
        "edit_timestamp": "2025-10-06T16:12:04.123456",
        # OR if edit skipped:
        "edit_skipped": True,
        "skip_reason": "No edit recommended — image is production-ready",
    }
}
```

#### Loop Control

- `auto_edit_count` tracks total auto-edit loop iterations (max 3)
- After editing, `recently_edited_asset_ids` is set (only actually-edited assets)
- Agent 6 re-reviews only those assets
- If no actual edits applied (all skipped) → routes to Agent 8 directly

#### Storage

- **State keys**: `Phase1State["edited_images"]`, `recently_edited_asset_ids`, `auto_edit_count`
- **MongoDB**: `update_agent_output(agent_number=7, output={"edited_images": {...}, "edited_asset_ids": [...]})`

---

### Agent 8 — Variation Generator

| Field | Value |
|---|---|
| Class | `VariationGeneratorAgent` |
| File | `phase_1_agents/agent_8_variation_generator.py` |
| Model | `gemini-3.1-flash-image-preview` (Nano Banana) with reference image |
| LangGraph node | `agent_8_node` |
| Retry logic | Max 3 retries with exponential backoff (2, 4, 8 seconds) |

#### Input

| Key | Type | Source |
|---|---|---|
| `edited_images` | `Dict` | Agent 7 output (preferred) |
| `generated_images` | `Dict` | Agent 5 output (fallback) |
| `approved_asset_ids` | `Optional[List[str]]` | Normalized UUIDs; if set, only these assets get variations |

Image data fallback chain: `edited_images` → `generated_images` → `state["generated_images"]` → MongoDB load → Agent 5 output file.

#### Character Variations (5 angles)

| Angle name | Description |
|---|---|
| `close_up` | Tight head-and-shoulders portrait, chest-level crop, f/1.8 |
| `wide_shot` | Full body, head-to-toe with breathing room |
| `profile_left` | Clean left-side profile |
| `profile_right` | Clean right-side profile |
| `back_shot` | Full body rear view |

#### Location Variations (4 directions via 2×2 grid)

Single Gemini call generates a 2×2 grid image; Python/Pillow crops it:

```
Grid layout (pixel coordinates):
  [North view]  [East view]       top-left=(0,0,W/2,H/2)    top-right=(W/2,0,W,H/2)
  [West view]   [South view]      bottom-left=(0,H/2,W/2,H) bottom-right=(W/2,H/2,W,H)
```

Saved as individual directional images: `north`, `east`, `west`, `south`

#### Prop Variations

None — props use master image only (`self.prop_angles = []`)

#### Image Generation Call

```python
response = gemini_client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[variation_prompt, reference_PIL_image],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="1:1")
    )
)
```

#### S3 Upload

```python
s3_key = f"phase1/variations/{timestamp}_{filename}"
# e.g. phase1/variations/20251006_161304_BLACK_LAB_PUPPY_close_up_1.png
```

Local path saved (for fallback): `output/variations/{timestamp}/{asset_type}/{safe_name}/{safe_name}_{angle}_{n}.png`

#### Output Schema

```python
Phase1State["variation_images"] = {
    "characters": [{
        "id": "uuid",
        "name": "ASSET NAME",
        "master_image": "s3_presigned_url_or_local_path",
        "variations": {
            "close_up": {
                "angle_description": "Head and shoulders close-up",
                "prompt": "Using the character in the reference image...",
                "images": [{"index": 1, "url": "s3_presigned_url", "local_path": "...", "s3_url": "...", "filename": "NAME_close_up_1.png"}]
            },
            "wide_shot": {...},
            "profile_left": {...},
            "profile_right": {...},
            "back_shot": {...}
        },
        "generation_timestamp": "2025-10-06T16:13:04.123456"
    }],
    "locations": [{
        "id": "uuid",
        "name": "LOCATION NAME",
        "master_image": "...",
        "variations": {
            "north": {"angle_description": "Northern view of the location", "prompt": "...", "images": [{"dimensions": "512x512", ...}]},
            "east": {...},
            "west": {...},
            "south": {...}
        },
        "generation_timestamp": "..."
    }],
    "props": []
}
```

#### Storage

- **State key**: `Phase1State["variation_images"]`
- **MongoDB**: `update_agent_output(agent_number=8, output={"variation_images": {...}})`

---

## 3. Phase 2: Shot Image Pipeline

### Workflow Overview

| File | `phase_2_agents/langgraph_workflow.py` |
|---|---|
| State type | `Phase2State` (TypedDict, `workflow_state.py`) |
| Entry point | `agent_1_strategy_node` |
| Orchestration | LangGraph `StateGraph` |

#### Phase 2 Agent Flow

```
Agent 1 (Shot Strategy)
    └─► Human Approval Checkpoint
            └─► Agent 2 (Image Prompt Generator)
                    └─► Agent 3 (Prompt Review)
                            └─► Agent 12 (Shot Design)
                                    └─► Agent 13 (Prompt Modifier)
                                            └─► Agent 14 (Imagen Generator — Initial)
                                                    └─► Agent 15 (Image Reviewer)
                                                            ├─► Agent 15A (Prompt Regeneration)
                                                            │       └─► Agent 14 (Regen) ─► Agent 15 (loop, max 3/shot)
                                                            ├─► Agent 7 (Shot Editor) ─► Agent 15 (loop, max 3/shot)
                                                            └─► Agents 16 / 17 / 18 (Product Fidelity Loop)
                                                                    └─► Final Approval Checkpoint
```

#### Phase 2 State Schema (`Phase2State`)

Key fields:

```python
class Phase2State(TypedDict):
    shot_list_request: Dict           # Raw shot list from API {episode_id, title, shots: [...]}
    show_id: str                      # Matches _id in production_projects
    episode_number: int
    project_id: Optional[str]         # Phase 1 project for asset loading
    movie_id: str                     # For visual_style lookup in movies collection
    episode_id: str
    title: str
    job_id: Optional[str]
    annotated_shot_list: AnnotatedShotList   # Set by Agent 1; updated through pipeline
    strategy_approval_decision: Optional[bool]
    image_prompts_generated: Dict            # Agent 2 output
    reviewed_prompts: Dict                   # Agent 3 output
    shot_designs: Dict                       # Agent 12 output
    modified_prompts: Dict                   # Agent 13 output
    generated_images: Dict                   # Agent 14 output
    image_reviews: Dict                      # Agent 15 output
    regenerated_prompts: Dict                # Agent 15A output
    edited_shots: Dict                       # Agent 7 (P2) output
    product_review_results: Dict             # Agent 16 output
    product_fix_prompts: Dict[str, str]      # Agent 17 output (shot_id → prompt)
    product_corrected_images: Dict[str, str] # Agent 18 output (shot_id → S3 URL)
    # Loop tracking
    edit_loop_iterations: Dict[str, int]     # shot_id → edit attempt count
    regenerate_loop_iterations: Dict[str, int]  # shot_id → regen attempt count
    product_review_iterations: Dict[str, int]   # shot_id → product fix attempt count
    shots_needing_edit: List[str]
    shots_needing_regeneration: List[str]
    shots_product_approved: List[str]
    shots_needing_product_fix: List[str]
    final_approval_decision: Optional[bool]
    mongodb_client: Optional[Any]            # ShotsService singleton
    product_image_url: Optional[str]         # Fetched from v1 project document
    pipeline_status: str
    current_agent: str
```

---

### Agent 1 (P2) — Shot Strategy

| Field | Value |
|---|---|
| Class | `ShotStrategyAgent` |
| File | `phase_2_agents/agent_shot_strategy/shot_strategy_agent.py` |
| Model | `ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.1)` via LangChain |
| LangGraph node | `agent_1_strategy_node` |

#### Input

```python
Phase2State["shot_list_request"]["shots"] = [{
    "shot_id": "S01E01_001",
    "description": "Wide establishing shot of the lakeside park...",
    "duration": 5.0,
    "scene_number": 1,
    "sequence_number": 1,
    "shot_style": "wide",
    "camera_movement": "static",
    "source_type": "generated",
    "characters": ["BLACK_LAB_PUPPY"],
    "locations": ["LAKESIDE_PARK"],
    "product_present": False
}]
```

#### Output

Each shot in `AnnotatedShotList.annotated_shots` gains a `generation_strategy`:

| Strategy | When used |
|---|---|
| `generate_new` | Standard shot; image generated from scratch |
| `last_frame_seed` | Shot follows a continuous action; previous shot's last frame used as seed |
| `multi_shot` | Complex multi-element composition |

`AnnotatedShotList` also carries `overall_continuity_notes` and `strategy_summary`.

#### Storage

- **MongoDB (shots collection)**: `mongodb_client.save_annotated_shots_to_atlas(annotated_shots, show_id, episode_number, episode_id, ...)`
- **State key**: `Phase2State["annotated_shot_list"]`, `strategy_analysis_results`
- **MongoDB (projects)**: `update_agent_output(agent_number=1)` *(skipped if already completed to avoid duplicates)*

#### Human Approval Checkpoint

Workflow pauses; `Phase2State["strategy_approval_decision"]: Optional[bool]` must be set via API:
- `True` → routes to Agent 2
- `False` → `pipeline_status = "rejected"`

---

### Agent 2 (P2) — Image Prompt Generator

| Field | Value |
|---|---|
| Class | `ImagePromptGeneratorAgent` |
| File | `phase_2_agents/image_prompt_generator_agent.py` |
| Model | `ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.3)` via LangChain |
| LangGraph node | `agent_2_prompt_generator_node` |

#### Input

| Key | Type | Source |
|---|---|---|
| `annotated_shot_list` | `AnnotatedShotList` | Agent 1 output |
| `AssetLibrary` | object | Constructed from MongoDB; fetches Phase 1 Agent 5/8 images for character, location, prop references |
| `product_shot_ids` | `Set[str]` | Shot IDs where `product_present=True` |
| `product_image_url` | `Optional[str]` | Fetched from v1 project |
| `scene_description` | `Optional[str]` | Episode-level scene context |

#### AssetLibrary

Queries Phase 1 outputs in `production_projects` (`agent5_output.generated_images`, `agent8_output.variation_images`) by `show_id`/`project_id`. Returns character, location, prop images as `AssetInfo` objects with their S3 URLs.

#### Processing

Async `generate_prompts_for_shots()` — for each shot, the LLM receives:
- Shot description + strategy
- Relevant asset reference images (from AssetLibrary)
- Product image (for product shots)
- Scene context

#### Storage

```python
# Per shot, saves v0 (initial draft) to shots collection
mongodb_client.update_shot_image_version(
    show_id=show_id,
    episode_number=episode_number,
    shot_id=shot.shot_id,
    version="v0",
    updated_prompt=v0_data["updated_prompt"],
    changes_made="Initial image prompt generated by Agent 2",
    reasoning="AI-generated prompt based on shot description and strategy",
    generated_images_s3=[]  # No images yet, just prompt
)
```

- **State key**: `Phase2State["image_prompts_generated"]`
- **MongoDB (projects)**: `update_agent_output(agent_number=2, output={"prompts": [...], "statistics": {...}})`

---

### Agent 3 (P2) — Prompt Review

| Field | Value |
|---|---|
| Class | `PromptReviewAgent` |
| File | `phase_2_agents/agent_prompt_review/prompt_review_agent.py` |
| Model | `gemini-3.1-pro-preview` via LangChain |
| LangGraph node | `agent_3_prompt_review_node` |

#### Input

`image_prompts_generated` (Agent 2 output) + `AssetLibrary` + `product_shot_ids` + `product_image_url`

#### Processing

Async `review_prompts()` — reviews all shot prompts together for visual continuity (consistent lighting, character appearances, background consistency across sequential shots). Produces `v1` version of each shot's image prompt.

#### Storage

```python
mongodb_client.update_shot_image_version(
    ..., version="v1",
    updated_prompt=v1_data["updated_prompt"],
    changes_made="Prompt reviewed and refined by Agent 3 for continuity",
    reasoning="Review agent applied continuity fixes and improvements",
    generated_images_s3=[]
)
```

- **State key**: `Phase2State["reviewed_prompts"]`, `Phase2State["annotated_shot_list"]` (updated with v1 prompts)
- **MongoDB (projects)**: `update_agent_output(agent_number=3)`

---

### Agent 12 — Shot Design

| Field | Value |
|---|---|
| Class | `ShotDesignAgent` |
| File | `phase_2_agents/shot_design_agent.py` |
| Model | No LLM — rule-based + feasibility scoring |
| LangGraph node | `agent_12_shot_design_node` |

#### Input

`annotated_shot_list` + `AssetLibrary` + `use_feasibility_check=True`

#### Processing

Per shot (`analyze_shot(shot, previous_shot_design)`):
1. Identifies characters and locations required by shot (from CSV `characters`/`locations` fields on the shot)
2. Looks up matching Phase 1 assets in `AssetLibrary`
3. Scores feasibility (0.0–1.0) based on asset availability
4. Recommends `composition_strategy` and `model_recommendation`
5. Detects warnings (missing assets, conflicting requirements)

#### Output per Shot (`ShotDesign` dataclass)

```python
@dataclass
class ShotDesign:
    shot_id: str
    generation_strategy: str
    selected_assets: List[Dict]        # [{name, type, url, variation_type}]
    composition_strategy: Dict
    model_recommendation: str
    feasibility_score: float           # 0.0–1.0
    warnings: List[str]
    metadata: Dict                     # {original_description, characters_found, scene_environment}
```

#### Storage

```python
shots_collection.update_one(
    {"show_id": show_id, "episode_number": ep_num, "annotated_shots.shot_id": shot_id},
    {"$set": {"annotated_shots.$.shot_design": design_dict}}
)
```

- **State key**: `Phase2State["shot_designs"]` — `{"designs": [design_dict, ...], "total_shots": N, "avg_feasibility": F}`
- **MongoDB (projects)**: `update_agent_output(agent_number=12, output={"shot_designs": [...], "statistics": {...}})`

---

### Agent 13 — Prompt Modifier

| Field | Value |
|---|---|
| Class | `PromptModifierAgent` |
| File | `phase_2_agents/prompt_modifier_agent.py` |
| Model | `gemini-3.1-pro-preview` via `google.genai` |
| LangGraph node | `agent_13_prompt_modifier_node` |

#### Input

| Key | Source |
|---|---|
| `shot_designs` | Agent 12 output |
| `AssetLibrary` | MongoDB Phase 1 assets |
| `visual_style` | Fetched from `movies` collection via `movie_id` → `global_settings.visual_style` |

#### Processing

Per shot (`modify_shot(shot_design, scene_baseline, is_product_shot)`):
1. Reads `warnings` from Agent 12's `ShotDesign`
2. Calls Gemini to produce a `corrected_prompt` that resolves those warnings
3. Lists `corrected_assets` (validated against AssetLibrary)
4. Computes `feasibility_change` (positive = improved)

**Product shot handling**: If `is_product_shot=True`, product image is flagged for injection by Agent 14.

#### Output per Shot (`ModifiedShot` dataclass)

```python
@dataclass
class ModifiedShot:
    shot_id: str
    corrected_prompt: str
    corrected_assets: List[Dict]       # [{name, type, url}]
    warnings_resolved: List[str]
    warnings_remaining: List[str]
    feasibility_change: float
    metadata: Dict
```

#### Storage

```python
shots_collection.update_one(
    {"show_id": identifier, "episode_number": ep_num, "annotated_shots.shot_id": shot_id},
    {"$set": {"annotated_shots.$.prompt_modifications": modified_dict}}
)
```

- **State key**: `Phase2State["modified_prompts"]` — `{"modified_shots": [{shot_id, corrected_prompt, corrected_assets, ...}], "total_shots", "total_warnings_resolved", "avg_feasibility_improvement"}`
- **MongoDB (projects)**: `update_agent_output(agent_number=13)`
- **Routing**: Auto-sets `prompt_approval_decision=True` (bypasses the prompt approval checkpoint)

---

### Agent 14 — Imagen Generator (Initial)

| Field | Value |
|---|---|
| Class | `ImagenGeneratorAgent` |
| File | `phase_2_agents/imagen_generator_agent.py` |
| Model | Gemini Imagen 4.0 (via `google.genai`) |
| LangGraph node | `agent_14_imagen_generator_node` |

#### Input

| Key | Source |
|---|---|
| `modified_prompts.modified_shots` | Agent 13 output |
| `movie_id` | For aspect ratio lookup |
| `product_image_url` + `product_present` shots | Product image injected into `corrected_assets` for relevant shots |

#### Product Image Injection

```python
product_asset_entry = {"name": "PRODUCT", "type": "product", "url": product_image_url}
# Prepended to shot["corrected_assets"] for each shot in product_shot_ids
```

#### S3 Upload (Initial Path)

```python
s3_key = f"phase2/{show_id}/{shot_id}/v0.png"
s3_url = s3_client.generate_presigned_url("get_object",
    Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=604800)
```

#### MongoDB Storage

```python
mongodb_client.update_shot_image_version(
    show_id=show_id, episode_number=ep_num, shot_id=shot_id,
    version="v0",
    updated_prompt=corrected_prompt,
    changes_made="Initial Imagen generation (Agent 14)",
    reasoning="Generated from Agent 13 corrected prompt",
    generated_images_s3=[s3_url]
)
```

#### Output

```python
Phase2State["generated_images"] = {
    "generated_images": [{
        "shot_id": "S01E01_001",
        "s3_url": "https://s3...v0.png",
        "prompt": "corrected prompt text",
        "local_path": "...",
        "image_version": "v0",
        "metadata": {...}
    }],
    "failed_generations": [...]
}
```

- **MongoDB (projects)**: `update_agent_output(agent_number=14, output={generated_images, failed_generations, statistics})`

---

### Agent 15 — Image Reviewer (Phase 2)

| Field | Value |
|---|---|
| Class | `ImageReviewAgent` |
| File | `phase_2_agents/image_reviewer_agent.py` |
| Model | `gemini-3.1-pro-preview` (vision) |
| LangGraph node | `agent_15_image_reviewer_node` |

Reviews each generated shot image against the shot description and prompt. Output per shot:

```python
review = {
    "shot_id": "S01E01_001",
    "decision": "approved" | "needs_edit" | "regenerate",
    "score": 0–100,
    "issues_found": [...],
    "edit_instructions": "...",
    "approved": bool
}
```

Routes shots to:
- `shots_needing_edit` → Agent 7 (P2) edit loop (max 3 iterations/shot)
- `shots_needing_regeneration` → Agent 15A regen loop (max 3 iterations/shot)
- Approved shots → Product fidelity check (Agents 16/17/18) or Final Approval

---

### Agent 15A — Prompt Regeneration

| Field | Value |
|---|---|
| Class | `PromptRegenerationAgent` |
| File | `phase_2_agents/agent_15A/prompt_regeneration_agent.py` |
| Model | `gemini-3.1-pro-preview` |
| LangGraph node | `agent_15A_prompt_regeneration_node` |

#### Input

| Key | Source |
|---|---|
| `shots_needing_regeneration` | Agent 15 routing |
| Agent 15 review data (`edit_instructions`, `issues_found`) | Per-shot feedback |
| Agent 13 prompts (`corrected_prompt`) | Original corrected prompt for context |
| Current S3 URL | From state or MongoDB (`image.v{latest}` or `edited_image_s3.v{latest}`) |

**Max iterations per shot**: 3. Shots that have already hit 3 iterations are skipped.

#### Processing

`regenerate_prompts_batch(shots_to_regenerate, shot_designs, modified_prompts, product_shot_ids)` — Gemini rewrites the prompt using review feedback as guidance.

#### Output

```python
Phase2State["regenerated_prompts"] = {
    "regenerated_shots": [{
        "shot_id": "S01E01_001",
        "updated_prompt": "revised prompt addressing the issues...",
        "reasoning": "The original prompt failed to...",
        "analysis": "..."
    }]
}
```

Routes to **Agent 14 (Regeneration)** which uploads to `v{n}.png` (next version).

---

### Agent 14 Regen — Imagen Generator (Regeneration)

Same class (`ImagenGeneratorAgent`), different LangGraph node (`agent_14_regenerate_node`).

**Key differences from initial Agent 14:**

1. Reads prompts from `regenerated_prompts.regenerated_shots` (Agent 15A output)
2. Fetches `corrected_assets` and `metadata` from Agent 13's original data
3. Auto-determines next version number from MongoDB
4. S3 key: `phase2/{show_id}/{shot_id}/v{next_version}.png`
5. **Merges** new images into existing `generated_images` (does NOT overwrite approved shots):
   ```python
   new_shot_ids = {img["shot_id"] for img in new_list}
   merged_list = [img for img in existing_list if img["shot_id"] not in new_shot_ids] + new_list
   ```

Routes back to Agent 15 for re-review.

---

### Agent 7 (P2) — Shot Editor

| Field | Value |
|---|---|
| Class | `ShotEditorAgent` |
| File | `phase_2_agents/agent_7_shot_editor.py` |
| Model | `gemini-3.1-flash-image-preview` (Nano Banana) |
| LangGraph node | `agent_7_shot_editor_node` |

Applies targeted image edits to shots marked `needs_edit` by Agent 15. Uses the same Nano Banana edit pattern as Phase 1 Agent 7.

`edit_loop_iterations: Dict[str, int]` tracks per-shot attempt count (max 3). Routes back to Agent 15 for re-review after each edit.

Storage: `edited_image_s3` versioned dict in `shots` collection per shot.

---

### Agents 16/17/18 — Product Fidelity Loop

These three agents run only when `product_present=True` shots exist AND `product_image_url` is set.

| Agent | Class | File | Purpose |
|---|---|---|---|
| 16 | `ProductReviewerAgent` | `agent_16_product_reviewer/product_reviewer_agent.py` | Reviews product fidelity vs uploaded image |
| 17 | `ProductPromptGenAgent` | `agent_17_product_prompt_gen/product_prompt_gen_agent.py` | Generates Nano Banana fix prompt |
| 18 | `ProductEditorAgent` | `agent_18_product_editor/product_editor_agent.py` | Applies fix via Nano Banana |

**State tracking**:
```python
product_review_iterations: Dict[str, int]   # shot_id → attempt count (max 3)
shots_needing_product_fix: List[str]        # shots failing product review
shots_product_approved: List[str]           # shots passing review (or force-passed at limit)
product_corrected_images: Dict[str, str]    # shot_id → latest S3 URL after correction
```

After this loop: **Final Approval Checkpoint** — `final_approval_decision: bool` set externally via API → `True` marks pipeline complete.

---

## 4. Phase 3: Video Generation Pipeline

### Workflow Overview

| File | `phase_3_agents/langgraph_workflow.py` |
|---|---|
| State type | `Phase3State` (`workflow_state.py`) |
| Invocation | One pipeline call **per shot** via `run_phase3_pipeline(shot_id, show_id, ...)` |
| Entry point | `initialize_node` |

#### Phase 3 Flow

```
initialize_node
    └─► prompt_router_node (generate video prompt)
            └─► video_generation_node (Veo 3.1)
                    └─► ai_review_node
                            ├─► video_generation_node (if refine/regenerate, max 3x)
                            └─► human_checkpoint_node
                                    ├─► video_generation_node (if needs_changes, max 3x)
                                    └─► END (if approved)
```

#### Phase 3 State Schema (`Phase3State`)

```python
class Phase3State(TypedDict):
    shot_id: str
    show_id: str
    episode_number: int
    shot_data: Dict[str, Any]              # Fetched from MongoDB shots collection
    image_version: Optional[str]           # "v0"|"v1"|"v2" — which shot image to use
    job_id: Optional[str]
    scene_number: Optional[int]
    sequence_number: Optional[int]
    generation_strategy: str               # "generate_new"|"last_frame_seed"|"multi_shot"
    video_prompt: str                      # Generated by prompt router
    prompt_version: str                    # "A" or "B"
    start_image_url: str                   # S3 URL of shot image
    video_generation_task_id: Optional[str]
    generated_video_url: Optional[str]     # S3 presigned URL
    video_generation_status: str
    video_generation_attempt: int          # Max 3 (AI-triggered)
    max_video_generation_attempts: int
    review_result: Optional[Dict]          # VideoReviewResult
    review_decision: Optional[str]         # "approved"|"refine_prompt"|"regenerate"
    review_score: Optional[int]
    suggested_prompt: Optional[str]
    human_decision: Optional[str]          # "approved"|"needs_changes"
    human_updated_prompt: Optional[str]
    human_feedback: Optional[str]
    human_regeneration_attempt: int        # Max 3
    max_human_regeneration_attempts: int
    current_node: str
    pipeline_status: str
    mongodb_save_status: str
    video_versions: List[Dict]             # All versions generated
    current_version: int                   # Increments each generation
```

---

### Initialize Node

**Function**: `initialize_node` in `langgraph_workflow.py`

Fetches shot document from MongoDB using a three-priority fallback:

| Priority | Location | Query |
|---|---|---|
| 1 | `shots` collection, `annotated_shots` array | `{"show_id": show_id, "annotated_shots.shot_id": shot_id}` |
| 2 | `shots` collection, individual document | `{"shot_id": shot_id, "show_id": show_id}` |
| 3 | `production_projects` fallback | `agent14.output.generated_images[shot_id]` + `agent12.output.shot_designs` |

**Resume detection**: If `state["human_decision"]` is set, workflow is resuming from a human checkpoint — re-fetches shot data and routes directly to `human_checkpoint_node`.

**Sets**: `generation_strategy`, `scene_number`, `sequence_number` (parsed from `shot_id` format `S{show}E{scene:02d}_{seq:03d}` as fallback)

---

### Prompt Router Node

**Function**: `prompt_router_node`

Selects prompt generation agent based on `generation_strategy`:

| Strategy | Agent Class | File | Prompt version |
|---|---|---|---|
| `generate_new` or `last_frame_seed` | `VideoGenerationAgent` | `video_prompt_A/agent_video_generation.py` | `"A"` |
| `multi_shot` | `MultiShotVideoGenerator` | `video_prompt_B/video_prompt_B.py` | `"B"` |

Both agents use `gemini-3.1-pro-preview` to write a cinematic video generation prompt from shot context.

**Storage**: `project_service.update_agent_output(agent_number=17, output={video_prompt, prompt_version, generation_strategy, shot_id})`

---

### Video Generation Node

| Field | Value |
|---|---|
| Class | `VideoGenerationAPIAgent` |
| File | `phase_3_agents/video_generation/video_generation_api_agent.py` |
| Model | **Google Veo 3.1** |
| LangGraph node | `video_generation_node` |

#### Input

| Key | Source |
|---|---|
| `video_prompt` | Prompt router (original), or `human_updated_prompt`, or `suggested_prompt` (from AI review) |
| `start_image_url` | Fetched from `shots` collection by `image_version` (e.g. `v0`, `v1`) |
| `shot_data` | MongoDB shot document |
| `scene_number`, `sequence_number` | For S3 naming |

**Prompt priority**: `human_updated_prompt` > `suggested_prompt` (if attempt > 0) > original `video_prompt`

#### S3 Key

```
phase3/{show_id}/S{show_num}E{scene:02d}_{seq:03d}/v{current_version}.mp4
# e.g. phase3/abc123/S01E01_001/v0.mp4
```

#### MongoDB Storage (two-step upsert)

```python
# Step 1: Initialize video field from null → {} (required for nested $set)
shots_collection.update_one(query,
    {"$set": {"annotated_shots.$[elem].video": {}}},
    array_filters=[{"elem.shot_id": shot_id, "elem.video": None}]
)

# Step 2: Set the version
shots_collection.update_one(query,
    {"$set": {
        f"annotated_shots.$[elem].video.{version}": {
            "updated_prompt": video_prompt,
            "changes_made": "...",
            "reasoning": "...",
            "generated_videos_s3": [video_url],
            "task_id": task_id,
            "timestamp": "...",
            "last_frame_s3": last_frame_url,   # optional
            "approval_status": "pending",
            "approval_feedback": "",
            "approved_at": None
        },
        "annotated_shots.$[elem].updated_at": datetime.now()
    }},
    array_filters=[{"elem.shot_id": shot_id}]
)
```

**Version tracking**: `current_version` increments each attempt; version string = `f"v{current_version}"`

**Video versions list** tracked in state:
```python
Phase3State["video_versions"] = [{
    "updated_prompt": "...",
    "changes_made": "...",
    "reasoning": "Video generation attempt N",
    "generated_videos_s3": ["s3_url"],
    "task_id": "...",
    "timestamp": "...",
    "last_frame_s3": "s3_url"   # if available
}]
```

- **MongoDB (projects)**: `update_agent_output(agent_number=18, output={shot_id, video_url, prompt, task_id, ...})`

---

### AI Review Node

| Field | Value |
|---|---|
| Class | `VideoReviewAgent` |
| File | `phase_3_agents/video_generation/video_review_agent.py` |
| Model | `gemini-3.1-pro-preview` (video understanding) |
| LangGraph node | `ai_review_node` |

#### Input

| Key | Source |
|---|---|
| `generated_video_url` | Video generation node output |
| `video_prompt` | Original prompt |
| `shot_description` | `shot_data["description"]` |
| `generation_strategy` | From state |

#### Output Schema (`VideoReviewResult`, `video_review_models.py`)

```python
{
    "decision": "approved" | "refine_prompt" | "regenerate",
    "overall_score": 0–100,
    "assessment": {
        "strengths": List[str],
        "issues": List[str],
        ...
    },
    "prompt_suggestions": {
        "suggested_prompt": str,
        "reasoning": str
    },
    "timestamp": "..."
}
```

#### Routing

| Decision | Condition | Action |
|---|---|---|
| `approved` | — | → `human_checkpoint_node` |
| `refine_prompt` or `regenerate` | `attempt < max_attempts` | → `video_generation_node` with `suggested_prompt` |
| `refine_prompt` or `regenerate` | `attempt >= max_attempts (3)` | → `human_checkpoint_node` |
| unknown | — | → `human_checkpoint_node` |

**Storage**: `project_service.update_agent_output(agent_number=19, append_output=True)` — appended (not overwritten) since multiple shots run through Agent 19.

---

### Human Checkpoint Node

**Function**: `human_checkpoint_node`

Workflow pauses with `pipeline_status = "waiting_for_human"`. Resume by calling `run_phase3_pipeline()` again with:
- `human_decision="approved"` → `pipeline_status = "completed"`
- `human_decision="needs_changes"` + `human_updated_prompt="..."` → back to `video_generation_node`

Max human regeneration attempts: 3 (`human_regeneration_attempt` counter)

When resuming, `shot_data` is re-fetched from MongoDB (not carried from prior run since it's passed as `{}`).

---

## 5. Cross-Cutting Infrastructure

### S3 Key Patterns

| Phase | Content | Pattern |
|---|---|---|
| Phase 1 | Master images | `phase1/generated_images/{YYYYMMDD_HHMMSS}_{SAFE_NAME}_{n}.png` |
| Phase 1 | Variations | `phase1/variations/{YYYYMMDD_HHMMSS}_{SAFE_NAME}_{angle}_{n}.png` |
| Phase 2 | Shot images (initial) | `phase2/{show_id}/{shot_id}/v0.png` |
| Phase 2 | Shot images (regen) | `phase2/{show_id}/{shot_id}/v{n}.png` |
| Phase 3 | Videos | `phase3/{show_id}/S{show}E{scene:02d}_{seq:03d}/v{n}.mp4` |

All S3 objects use **presigned URLs** with 7-day expiry (`ExpiresIn=86400 * 7`).

### MongoDB Write Patterns

#### Production Projects (`production_projects`)

```javascript
// update_agent_output writes:
{
  "agent_outputs.agent{N}.status": "completed",
  "agent_outputs.agent{N}.executed_at": ISODate(),
  "agent_outputs.agent{N}.output": { /* agent data */ },
  "updated_at": ISODate()
}
```

#### Shots Collection (`shots`)

Image versioning:
```javascript
// update_shot_image_version writes to annotated_shots[shot_id].image.v{N}:
{
  "updated_prompt": "...",
  "changes_made": "...",
  "reasoning": "...",
  "generated_images_s3": ["presigned_url"],
  "timestamp": ISODate()
}
```

Video versioning (two-step):
```javascript
// Step 1: initialize null → {}
// Step 2: write video.v{N}:
{
  "updated_prompt": "...",
  "generated_videos_s3": ["presigned_url"],
  "task_id": "...",
  "last_frame_s3": "presigned_url",
  "approval_status": "pending",
  "approval_feedback": "",
  "approved_at": null
}
```

#### Production Pipelines (`production_pipelines`)

Job status updates:
```javascript
{
  "agent{N}_status": "running" | "completed" | "failed" | "skipped" | "retrying",
  "current_agent": "agent_{N}",
  "pipeline_status": "running",
  "updated_at": ISODate()
}
```

---

## 6. Naming Conventions & Nomenclature

### Asset IDs

- **Format**: UUID v4 string, e.g. `"3f2504e0-4f89-11d3-9a0c-0305e82c3301"`
- **Generated by**: Pydantic `Field(default_factory=lambda: str(uuid.uuid4()))` in Agent 1 models
- **Propagated through**: All Phase 1 agents; carried unchanged through Agents 2–8
- **Used as**: Lookup key for image storage, review routing, regeneration targeting

### Asset Names

- **Format**: UPPERCASE WITH SPACES as extracted from script, e.g. `"BLACK LAB PUPPY"`, `"LAKESIDE PARK"`
- **When used as filename/S3 key**: Spaces and `/` replaced with `_`, e.g. `"BLACK_LAB_PUPPY"`
- **Normalization utility**: `backend.services.production.app.utils.name_normalization.normalize_asset_name()` — used for CSV comparison

### Shot IDs

- **Format**: `S{show_num:02d}E{scene:02d}_{seq:03d}` (but only last segment appears in practice)
- **Example**: `S01E01_001` (scene 1, shot 1 in scene), `S01E03_007`
- **Parsed by**: `parse_scene_sequence_from_shot_id()` regex `^S\d+E(\d+)_(\d+)`

### Image Version Keys

| Key | Meaning |
|---|---|
| `v0` | Initial generation (Agent 2 prompt, Agent 14 generation) |
| `v1` | Post-review revision (Agent 3 prompt refinement, Agent 14 regen or Agent 7 edit) |
| `v2`, `v3`, ... | Subsequent regeneration or edit iterations |

### S3 Filename Safe Names

```python
safe_name = asset_name.replace(" ", "_").replace("/", "_")
# e.g. "BLACK LAB PUPPY" → "BLACK_LAB_PUPPY"
# e.g. "THE LAKE (BACKGROUND ELEMENT)" → "THE_LAKE_(BACKGROUND_ELEMENT)"
```

### Pipeline Status Values

| Value | Meaning |
|---|---|
| `pending` | Not yet started |
| `running` | Active processing |
| `completed` | Successfully finished |
| `failed` | Error occurred |
| `waiting_for_approval` | Paused at human checkpoint (Phase 2 strategy) |
| `waiting_for_prompt_approval` | Paused at prompt checkpoint (Phase 2) |
| `waiting_for_final_approval` | Paused at final checkpoint (Phase 2) |
| `waiting_for_human` | Paused at human checkpoint (Phase 3) |
| `rejected` | Human rejected at checkpoint |
| `generating_variations` | Agent 8 active |
| `regenerating_images` | Regeneration loop active |

### Agent Status Values (per-agent in job record)

`pending` → `running` → `completed` | `failed` | `skipped` | `retrying`

### Review Decision Values

| Phase | Values |
|---|---|
| Phase 1 image review (Agent 6) | `approved`, `needs_edit`, `regenerate` |
| Phase 2 image review (Agent 15) | `approved`, `needs_edit`, `regenerate` |
| Phase 3 video review (Agent 19) | `approved`, `refine_prompt`, `regenerate` |
| Phase 3 human checkpoint | `approved`, `needs_changes` |

### Generation Strategy Values (Phase 2/3)

| Value | Description |
|---|---|
| `generate_new` | Standard generation from prompt |
| `last_frame_seed` | Uses previous shot's last frame as seed image |
| `multi_shot` | Multi-element composition; uses Prompt B |

### Visual Style Values (Phase 1/2)

| Value | Prompt prefix required |
|---|---|
| `realistic` | `"Raw, unretouched photograph"` |
| `pixar` | `"Pixar-style 3D animation"` |
| `2d` | `"2D animation style"` |

### Routing Key Format for Asset-Level Operations

Used in `needs_editing_assets` and `needs_regeneration_assets`:
```
"{asset_type_plural}:{asset_uuid}"
# e.g. "characters:3f2504e0-4f89-11d3-9a0c-0305e82c3301"
# e.g. "locations:7d58fa12-aaaa-bbbb-cccc-000000000001"
# e.g. "props:abcdef12-1234-5678-90ab-cdef01234567"
```

Edit key format in `edited_images` dict:
```
"{asset_type}:{asset_id}:image_{image_index}"
# e.g. "characters:3f2504e0...:image_1"
```

### Loop Caps

| Loop | Cap | State key tracking |
|---|---|---|
| Phase 1 auto-edit (Agent 7 → Agent 6) | 3 | `auto_edit_count` |
| Phase 1 auto-regen (Agent 6 → Agent 5) | 3 | `auto_regeneration_count` |
| Phase 1 human regen limit | 5 | `max_regenerations` |
| Phase 1 Agent 8 retry | 3 | `agent8_retry_count` |
| Phase 2 edit loop per shot | 3 | `edit_loop_iterations[shot_id]` |
| Phase 2 regen loop per shot | 3 | `regenerate_loop_iterations[shot_id]` |
| Phase 2 product fidelity per shot | 3 | `product_review_iterations[shot_id]` |
| Phase 3 AI-triggered regen | 3 | `video_generation_attempt` vs `max_video_generation_attempts` |
| Phase 3 human-triggered regen | 3 | `human_regeneration_attempt` vs `max_human_regeneration_attempts` |
