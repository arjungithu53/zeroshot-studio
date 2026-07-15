# Zeroshot Studio — Product Documentation

**Version:** 1.1  
**Last Updated:** 2026-05-28

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [System Architecture](#2-system-architecture)
3. [Pre-Production Pipeline](#3-pre-production-pipeline)
4. [Production Pipeline](#4-production-pipeline)
5. [API Reference](#5-api-reference)
6. [Data Models](#6-data-models)
7. [Infrastructure & Deployment](#7-infrastructure--deployment)
8. [Environment Variables](#8-environment-variables)
9. [AI Models](#9-ai-models)
10. [Quota & Rate Limiting](#10-quota--rate-limiting)

---

## 1. Product Overview

Zeroshot Studio is an **end-to-end AI video production platform** that transforms a brand brief or script into fully generated video content — without a human touching a camera.

### What It Does

| Input | Output |
|-------|--------|
| Brand brief + product description | Strategic positioning & audience insights |
| Script + shotlist CSV | Production-ready asset library (characters, locations, props) |
| Asset library + shot descriptions | Composition-aware shot images |
| Shot images | Finished video clips per shot |

### Key Value Propositions

- **Zero manual production:** Script → video with AI handling asset generation, image composition, and video synthesis
- **Human-in-the-loop checkpoints:** Approval gates at critical stages (asset library, shot prompts, final images, video)
- **Master pipeline mode:** Single API call triggers all phases end-to-end with auto-approval for unattended operation
- **Versioned everything:** Every image and video is version-tracked (v0 → v1 → v2) to support iterative editing and A/B comparison
- **Product fidelity:** Dedicated product review pipeline ensures uploaded product images appear faithfully in generated shots

### Two Services

| Service | Role |
|---------|------|
| **Pre-Production** | Strategic planning — transforms a brand brief into a production-ready script and shotlist through ~55 sequential AI agents across 3 phases |
| **Production** | Execution — transforms the script and shotlist into visual assets, shot images, and videos through 29+ AI agents across 3 phases |

---

## 2. System Architecture

### High-Level Stack

```
Client / Frontend
       │ HTTP REST
       ▼
FastAPI Service (port 8000)           — app/main.py
       │
       ├── Synchronous: project CRUD, status polls, approval gates
       └── Async: Celery task dispatch via AWS SQS
                  │
                  ▼
           Celery Workers             — app/tasks/*.py
                  │
          ┌───────┼───────┐
          ▼       ▼       ▼
       Phase 1  Phase 2  Phase 3     — LangGraph state machines
       (8 agents)(18 agents)(3 agents)
          │
          ├── MongoDB  (state + results)
          ├── AWS S3   (images + videos)
          └── Google Cloud AI APIs
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI + uvicorn |
| Async task queue | Celery + AWS SQS |
| Workflow orchestration | LangGraph (state machine per phase) |
| Database | MongoDB Atlas (6 collections) |
| File storage | AWS S3 |
| Cache / quota | Redis (DB 1) |
| AI — text & vision | Google Gemini Pro / Flash |
| AI — images | Google Imagen 4.0 |
| AI — video | Google Gemini Veo 3.1 |
| AI — image editing | SeeDream 4 Edit |
| Rate limiting | slowapi (per-user, Redis-backed) |

---

## 3. Pre-Production Pipeline

The pre-production service is a **stateless FastAPI app** running ~55 sequential AI agents across 3 phases. It produces the strategic foundation and final shotlist that feeds the production pipeline.

### Phase 1 — Strategic Foundation (13 Agents)

Transforms a company brief into strategic positioning.

| # | Agent | Output |
|---|-------|--------|
| 1 | Company Research | Market context, brand facts |
| 2 | Visual Context Extraction | Brand visual language |
| 3 | Brand Adjective Agent | Positioning adjectives |
| 4 | Audience Persona | Target audience profiles |
| 5 | Competitive Landscape | Differentiators vs. competitors |
| 6 | Central Human Truth | Core emotional hook |
| 7 | Value Prop & Offer | Offer and benefit framing |
| 8 | Truest Thing | Brand essence statement |
| 9 | Conflict Identification | Truth vs. obstacle mapping |
| 10 | Insight Validation | Validated audience insights |
| 11 | Truth Conflict Platform | Strategic narrative platform |
| 12 | Strategy Models | Framework-based strategy (Jobs-to-be-done, etc.) |
| 13 | Positioning Alignment | Brand-market fit confirmation |

### Phase 2 — Creative Development (25 Agents)

Generates and filters video concepts based on strategy.

| Group | Agents |
|-------|--------|
| Constraint management | Creative Constraint Manager → Constraint Priority Resolver → Idea Core Preservation → Brand Guideline Alignment |
| Video structure | Video Type Selection → Duration Structuring → Scene Deconstruction → Scene Role Enumeration → Scene Role Selector → Temporal Placement Solver → Scene Integration Plan |
| Narrative | Narrative Skeleton Generator → Narrative Skeleton Planner → Narrative Archetype Selector → Platform Behavior Optimizer |
| Ideation | Pattern Interrupt Generator → Viral Mechanics → Mental Model Transformer → Intergalactic Thinking → Concept Generator |
| Concept filtering | Offer Narrative Integrator → Concept Categorization → Concept Diversity Controller → Interest Filter → Concept Kill Switch |

### Phase 3 — Audio-Visual Scripting (12 Agents)

Converts approved concepts into a production-ready shotlist with audio specs.

| # | Agent | Output |
|---|-------|--------|
| 1 | Beat to Timeline Mapper | Scene timing structure |
| 2 | Offer Integration Planner | CTA placement in timeline |
| 3 | Visual Sequencing | Shot order and transitions |
| 4 | AV Separation | Separate visual and audio tracks |
| 5 | Voiceover Writer | Voiceover script per beat |
| 6 | Dialogue Agent | On-screen dialogue |
| 7 | Audio Design | Sound design specifications |
| 8 | Rhythm & Pacing Regulator | Timing refinements |
| 9 | Loop Optimization | Loopable edits |
| 10 | Constraint Compliance QA | Final compliance check |
| 11 | Shot Level Script Formatter | Per-shot formatted script |
| 12 | Final Shotlist Agent | Final shotlist → **input to Production Phase 1** |

---

## 4. Production Pipeline

The production service runs as a **FastAPI app + Celery workers**. All heavy AI work is dispatched to Celery tasks via AWS SQS and executed asynchronously by workers.

### Phase 1 — Asset Generation

**Purpose:** Build a complete visual asset library from the script.  
**Scope:** Runs once per movie (all scenes share the asset library).  
**LangGraph workflow:** `app/services/phase_1_agents/langgraph_workflow.py`

#### Agent Flow

```
Script + Shotlist CSV + Visual Style + Product Image
          │
          ▼
Agent 1 — Asset Extractor
  Uses Gemini structured output to extract:
  • Characters (names, physical descriptions)
  • Locations (setting, time of day, atmosphere)
  • Props (objects, including product prop)
  Uses CSV shotlist as source-of-truth to constrain extraction.
          │
          ▼
Agent 2 — Asset Reviewer
  Validates completeness; CSV-filters phantom entities.
          │
          ▼
Agent 3 — Prompt Generator
  Writes image generation prompts per asset × visual style.
          │
          ▼
Agent 4 — Prompt Optimizer
  Refines prompts for Imagen 4.0 quality.
          │
          ▼
Agent 5 — Image Generator (Imagen 4.0)
  Generates one image per asset.
  Injects pre-uploaded product image for product props.
          │
          ▼
Agent 6 — Image Reviewer (Gemini Vision)
  Per-image decision: approve / needs_edit / regenerate
          │
    ┌─────┼─────────────┐
    ▼     ▼             ▼
approve  edit          regen
    │    │              │
    │    Agent 7        └→ Agent 5
    │    SeeDream Edit    (rewritten prompt)
    │    (max 3 loops     
    │    back to A6)      
    │
    ▼
Agent 8 — Variation Generator (Imagen)
  Generates 6 camera angles per character:
  back_shot, close_up, profile_right, profile_left, wide_shot, master
          │
          ▼
  ┌──────────────────────┐
  │  Human Checkpoint    │
  │  /phase1/approve     │  → approve
  │  /phase1/reject      │  → regenerate from Agent 5
  │  /phase1/retry-asset │  → retry single asset
  └──────────────────────┘
          │ approved
          ▼
  Assets saved to MongoDB
  production_projects.agent_outputs  (project workflow)
  assets_collections collection       (movie workflow)
```

#### Key Data Outputs

```
agent_outputs.agent8.variation_images
  characters[].variations
    back_shot:     { url, s3_url, local_path }
    close_up:      { url, s3_url, local_path }
    profile_right: { url, s3_url, local_path }
    profile_left:  { url, s3_url, local_path }
    wide_shot:     { url, s3_url, local_path }
    master:        { url, s3_url, local_path }

agent_outputs.agent5.generated_images
  characters[] / locations[] / props[]
    images[]: { url, s3_url, local_path, prompt }
```

---

### Phase 2 — Shot Image Generation

**Purpose:** Generate a composition-aware image for every shot using Phase 1 assets.  
**Scope:** Runs **per scene**; can run in parallel across scenes.  
**LangGraph workflow:** `app/services/phase_2_agents/langgraph_workflow.py`

#### AssetLibrary Bridge

`helpers/asset_library.py` is the critical link between Phase 1 and Phase 2. It loads Phase 1 outputs from MongoDB and exposes them as `AssetInfo` objects for agent prompting and shot design.

```python
AssetInfo:
    name: str
    type: str         # character | location | prop
    angle: str        # back_shot | close_up | profile_right | ...
    local_path: str
    url: Optional[str]
    prompt: str       # original generation prompt
    technical: str    # technical image specs
    framing: str      # framing/composition details
```

#### Agent Flow

```
Shot List + AssetLibrary
          │
          ▼
Agent 1 — Shot Strategy
  Per-shot assignment of generation_strategy:
  • generate_new       — text-to-image from scratch
  • last_frame_seed    — seed from previous shot's last frame
  • multi_shot         — image-to-video from first frame reference
          │
  ┌──────────────────────┐
  │  Human Checkpoint 1  │
  │  Approve Strategies  │
  └──────────────────────┘
          │ approved
          ▼
Agent 2 — Image Prompt Generator
  Cinematic prompts per shot using AssetLibrary.
          │
          ▼
Agent 3 — Prompt Reviewer
  QA + refinement pass on all prompts.
          │
          ▼
Agent 12 — Shot Design
  Asset selection + composition strategy per shot.
  Runs PromptFeasibilityChecker to flag unavailable assets.
          │
          ▼
Agent 13 — Prompt Modifier
  Adapt prompts to real asset availability.
          │
  ┌──────────────────────┐
  │  Human Checkpoint 2  │
  │  Approve Prompts     │
  └──────────────────────┘
          │ approved
          ▼
Agent 14 — Imagen Generator
  Google Imagen 4.0; version-tracked (v0, v1, v2…).
          │
          ▼
Agent 15 — Image Reviewer (Gemini Vision)
  Per-shot decision: approve / refine / regenerate / edit
          │
    ┌─────┼───────────┬──────────────────┐
    ▼     ▼           ▼                  ▼
approve  regen       edit            product_present?
    │   Agent 15A   Agent 7               │
    │   Rewrite →   SeeDream →      ┌─────┘
    │   Agent 14    Agent 15        │
    │   (max 3)     (max 3)         ▼
    │                          Agent 16 — Product Reviewer
    │                          Fidelity check on product shots
    │                               │
    │                         ┌─────┴─────┐
    │                         ▼           ▼
    │                       ok        needs fix
    │                         │           │
    │                         │    Agent 17 — Product Prompt Gen
    │                         │    Agent 18 — Product Editor
    │                         │    (max 3 loops back to Agent 16)
    │                         │
    ▼─────────────────────────┘
  ┌──────────────────────┐
  │  Human Checkpoint 3  │
  │  Final Image Approval│
  └──────────────────────┘
          │ approved
          ▼
  Shot images saved to MongoDB shots collection
  shots.annotated_shots[].image  (versioned v0, v1, v2…)
```

---

### Phase 3 — Video Generation

**Purpose:** Generate a video clip for every shot using its approved image as reference.  
**Scope:** Runs **per shot**; parallelised across shots after Phase 2 completes.  
**LangGraph workflow:** `app/services/phase_3_agents/langgraph_workflow.py`

#### Generation Strategies

| Strategy | Prompt Type | Veo API Call |
|----------|------------|--------------|
| `generate_new` | Video Prompt A — detailed cinematic description | Text-to-video |
| `last_frame_seed` | Video Prompt A — with last frame as seed image | Image-to-video (last frame) |
| `multi_shot` | Video Prompt B — ≤30 word consistent prompt | Image-to-video (first frame) |

#### Agent Flow

```
shot_id → Load shot data from MongoDB
          │
          ▼
    Check generation_strategy
          │
    ┌─────┴─────┐
    ▼           ▼
Video         Video
Prompt A      Prompt B
    │           │
    └─────┬─────┘
          ▼
  Gemini Veo 3.1 API
  (video_generation_api_agent.py)
  Poll until complete → upload to S3
          │
          ▼
Agent 19 — AI Video Review (Gemini Vision)
  Score: 0–100
  Decision: approved / refine_prompt / regenerate
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
approve refine regen
    │     │     └→ Veo (max 3)
    │     └→ Suggested prompt → Veo (max 3)
    │
  ┌──────────────────────┐
  │  Human Checkpoint    │
  │  Video Approval      │
  └──────────────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
approved   needs_changes
    │           └→ Human feedback prompt → Veo (max 3)
    ▼
  Video URL saved to MongoDB shots collection
  shots.annotated_shots[].video  (versioned)
```

---

### Master Pipeline

**Endpoint:** `POST /api/v1/master/run-pipeline`  
**Purpose:** Full end-to-end run with all human checkpoints **auto-approved**.

```
run_master_pipeline_task
       │
       ├── Start Phase 1 (all scenes share one asset library)
       │
       └── poll_phase1_task (polls until Phase 1 complete)
              │ auto-approve Phase 1 checkpoint
              │
              └── Start Phase 2 per scene (parallel)
                     │
                     └── poll_phase2_task
                            │ auto-approve all Phase 2 checkpoints
                            │
                            └── Start Phase 3 per shot (parallel)
                                   │
                                   └── poll_phase3_task
                                          │ auto-approve Phase 3 checkpoints
                                          │
                                          └── Pipeline complete
```

Status polling: `GET /api/v1/master/status/{master_job_id}`  
Returns per-scene and per-shot progress summary.

---

## 5. API Reference

Base URL: `http://localhost:8000`

### Health & System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info + endpoint list |
| GET | `/health` | MongoDB + S3 connectivity check (200 / 503) |
| GET | `/mongodb/connections` | Active connections + pool stats |

### Projects

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/projects` | List all projects |
| POST | `/api/v1/projects/create` | Create project with script → auto-starts Phase 1 |
| POST | `/api/v1/projects/create-name` | Create project (draft, name only) |
| POST | `/api/v1/projects/{project_id}/upload-files` | Upload script / shotlist / product image |
| GET | `/api/v1/projects/{project_id}` | Retrieve full project document |
| GET | `/api/v1/projects/{project_id}/status` | Project status + agent progress (0–7) |

### Phase 1 (Asset Generation)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/phase1/start` | Start Phase 1 Celery task |
| POST | `/api/v1/phase1/approve/{job_id}` | Approve assets at human checkpoint |
| POST | `/api/v1/phase1/reject/{job_id}` | Reject → trigger regeneration |
| POST | `/api/v1/phase1/retry-asset/{job_id}` | Retry single failed asset |
| GET | `/api/v1/phase1/status/{job_id}` | Job status |
| GET | `/api/v1/phase1/results/{job_id}` | Generated assets |
| GET | `/api/v1/phase1/task-status/{celery_task_id}` | Celery task status |

### Phase 2 (Shot Image Generation)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/phase2/start` | Start Phase 2 per-scene Celery task |
| POST | `/api/v1/phase2/approve-strategy/{job_id}` | Approve shot strategies (Checkpoint 1) |
| POST | `/api/v1/phase2/approve-prompts/{job_id}` | Approve corrected prompts (Checkpoint 2) |
| POST | `/api/v1/phase2/final-approve/{job_id}` | Final image approval (Checkpoint 3) |
| GET | `/api/v1/phase2/status/{job_id}` | Job status |
| GET | `/api/v1/phase2/results/{job_id}` | Download generated images (ZIP) |
| GET | `/api/v1/phase2/task-status/{task_id}` | Celery task status |
| GET | `/api/v1/phase2/mongodb/shots` | Raw MongoDB shots data |

### Phase 3 (Video Generation)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/phase3/start` | Start Phase 3 per-shot Celery task |
| POST | `/api/v1/phase3/approve/{job_id}` | Approve generated video |
| GET | `/api/v1/phase3/task-status/{task_id}` | Celery task status |

### Movies (Multi-Scene)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/movies/create` | Create movie + upload script/shotlist CSVs |
| GET | `/api/v1/movies/{movie_id}` | Get movie details |
| POST | `/api/v1/movies/{movie_id}/phase2-bootstrap` | Trigger Phase 2 for all scenes |
| POST | `/api/v1/movies/{movie_id}/phase3-bootstrap` | Trigger Phase 3 for all shots |

### Master (End-to-End)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/master/run-pipeline` | Start full pipeline (all checkpoints auto-approved) |
| GET | `/api/v1/master/status/{master_job_id}` | Poll overall pipeline status per scene/shot |

---

## 6. Data Models

### MongoDB Collections

#### `production_projects`

Stores per-project state and all Phase 1 agent outputs.

```
_id                      ObjectId
name                     string
script                   string (optional)
status                   draft | pending | extracting | prompting |
                         generating | reviewing | completed | failed
created_at / updated_at  timestamps
movie_id                 ref → movies._id
assets_collection_id     ref → assets_collections._id
scene_number             int
scene_script_s3_url      string
shotlist_json_s3_url     string
product_image_s3_url     string

agent_outputs:
  agent1 → extracted_assets  { characters[], locations[], props[] }
  agent2 → reviewed_assets
  agent3 → generated_prompts
  agent4 → optimized_prompts
  agent5 → generated_images  { characters[], locations[], props[] }
  agent6 → image_reviews
  agent7 → edited_images
  agent8 → variation_images  { characters[].variations { back_shot, close_up, … } }
```

#### `movies`

Stores multi-scene movie metadata.

```
_id                ObjectId
title              string
description        string
genre              string
assets_collection_id  ref → assets_collections._id
project_ids        ObjectId[]
total_scenes       int
total_characters   int
total_locations    int
total_props        int
created_at / updated_at

scenes[]:
  scene_number     int
  scene_name       string
  script           string
  shotlist         string
  project_id       ObjectId
  status           pending | phase1_complete | phase2_running |
                   phase2_complete | phase3_running | phase3_complete
```

#### `assets_collections`

Shared asset library for movie workflows (all scenes share one collection).

```
_id                ObjectId
movie_id           ref → movies._id
assets[]:
  name             string
  type             character | location | prop
  data             { description, features, … }
  images[]:
    version        string (v1, v2, …)
    url            string
    s3_url         string
    prompt         string
```

#### `shots`

Per-episode shot data for Phase 2 and Phase 3.

```
_id                ObjectId
show_id            string (project/show identifier)
episode_number     int

annotated_shots[]:
  shot_id          string (UUID)
  description      string
  duration         float (seconds)
  generation_strategy  generate_new | last_frame_seed | multi_shot

  shot_design:           Agent 12 output
  prompt_modifications:  Agent 13 output

  image:
    v0, v1, v2 …:  { url, s3_url, prompt, timestamp }
    selected:      string (human-selected version key)
    product_present: bool
    approval_status: pending | approved | needs_changes

  video:
    v0, v1 …:      { url, s3_url, prompt, review_score }
    approval_status: pending | approved | needs_changes
```

#### `production_pipelines`

Lightweight job tracker for each phase run.

```
_id                ObjectId
job_id             UUID (human-readable, used in API paths)
project_id         ObjectId
movie_id           ObjectId
assets_collection_id  ObjectId
shot_id            string
type               phase1_project | phase1_movie | phase2 | phase3
status             pending | running | waiting_for_human_approval |
                   completed | failed
celery_task_id     string
current_agent      string
agent{1..19}_status  pending | running | completed | failed
human_approval_decision  string
feedback           string
regeneration_count int
max_regenerations  int (default 3)
error_message      string
output_files       object
```

#### `production_assets`

Standalone asset documents (alternative to embedding in projects).

```
_id        ObjectId
name       string
type       character | location | prop
data       { description, features, is_product, … }
images[]   { version, url, s3_url, prompt }
```

---

## 7. Infrastructure & Deployment

### Startup

`start.sh` orchestrates both the API server and Celery workers:

```bash
# Validates all required env vars
# Starts uvicorn (FastAPI) on port 8000
# Starts Celery worker(s) listening to SQS queues
# Writes PIDs to pids/api.pid and pids/worker.pid
# Streams logs to logs/ directory
```

### Celery Configuration (`app/celery_app.py`)

| Setting | Value |
|---------|-------|
| Broker | AWS SQS (`production_WORKFLOW_QUEUE`) |
| Backend | Not used (results in MongoDB) |
| Task retries | 3 attempts, 60-second delay |
| Hard time limit | 2 hours |
| Soft time limit | 1 hour 55 minutes |
| Task routing | Phase 1 / 2 / 3 tasks → separate queues |

### MongoDB Connection (`app/config.py`)

- Singleton `MongoClientFactory` — connection pooled, reused across requests
- Database name: `production_MONGODB_DATABASE_NAME` env var
- 6 primary collections (see §6)

### Redis Usage

| Database | Purpose |
|----------|---------|
| DB 1 | Quota counters (atomic increment + TTL) |
| DB 1 | Idempotency keys (deduplication for retried requests) |

### S3 Organisation

```
{bucket}/
  assets/           ← Phase 1 character/location/prop images
  variations/       ← Phase 1 camera-angle variation images
  shots/            ← Phase 2 shot images
  videos/           ← Phase 3 video clips
  scripts/          ← Uploaded script files
  shotlists/        ← Uploaded shotlist CSVs
  product_images/   ← Uploaded product reference images
```

---

## 8. Environment Variables

### Service-Specific (`production_*`)

| Variable | Description |
|----------|-------------|
| `production_MONGODB_URI` | MongoDB Atlas connection string |
| `production_MONGODB_DATABASE_NAME` | Database name (e.g. `production`) |
| `production_AWS_ACCESS_KEY_ID` | S3 / SQS access key |
| `production_AWS_SECRET_ACCESS_KEY` | S3 / SQS secret key |
| `production_AWS_REGION` | AWS region (e.g. `us-east-1`) |
| `production_S3_BUCKET_NAME` | S3 bucket for images and videos |
| `production_WORKFLOW_QUEUE` | SQS queue name for Celery tasks |

### Shared (`SHARED_*`)

| Variable | Description |
|----------|-------------|
| `SHARED_GOOGLE_API_KEY` | Google Cloud AI API key (Gemini, Imagen, Veo) |
| `SHARED_REDIS_URL` | Redis connection URL |
| `SHARED_OPENAI_API_KEY` | OpenAI API key (optional fallback) |
| `SHARED_ANTHROPIC_API_KEY` | Anthropic Claude API key (optional) |

### Infrastructure (unprefixed)

| Variable | Description |
|----------|-------------|
| `AWS_ACCOUNT_ID` | AWS account ID (used for SQS URL construction) |
| `ENVIRONMENT` | `development` \| `staging` \| `production` |
| `DEBUG` | `true` \| `false` |
| `PORT` | API server port (default `8000`) |

---

## 9. AI Models

| Model | Provider | Used In | Purpose |
|-------|----------|---------|---------|
| **Gemini Pro** | Google | All phases | Structured text extraction, prompt generation, strategy reasoning |
| **Gemini Flash** | Google | Image & video review | Multimodal vision review (fast) |
| **Google Imagen 4.0** | Google Cloud Vertex AI | Phase 1 Agent 5, Phase 1 Agent 8, Phase 2 Agent 14 | Text-to-image generation, camera angle variations, shot images |
| **Gemini Veo 3.1** | Google Cloud | Phase 3 | Text-to-video, image-to-video (first frame or last frame seed) |
| **SeeDream 4 Edit** | BytePlus | Phase 1 Agent 7, Phase 2 Agent 7 & 18 | Targeted image editing based on review feedback |

### Generation Strategy → Model Mapping

```
Script/brief text
      └→ Gemini Pro (extraction + prompting)
            └→ Imagen 4.0 (asset images)
                  └→ Gemini Flash Vision (review)
                        └→ SeeDream Edit (corrections)
                              └→ Imagen 4.0 (variations)
                                    └→ Imagen 4.0 (shot images)
                                          └→ Veo 3.1 (video clips)
```

---

## 10. Quota & Rate Limiting

### User Quotas (`app/core/quota.py`)

Quotas are enforced atomically via Redis before any pipeline task is dispatched.

| Pipeline | Daily Limit | Weekly Limit |
|----------|-------------|--------------|
| `production_workflow` | 50 runs/user | 280 runs/user |

```python
# Before task dispatch
quota_manager.check_quota(user_id, "production_workflow", cost=1)
quota_manager.consume_quota(user_id, "production_workflow", cost=1)
```

### Idempotency (`app/core/idempotency.py`)

Every phase start request accepts an idempotency key header. If the same key is submitted twice (e.g., after a network timeout), the second request returns the original response without spawning a duplicate task.

```
generate_idempotency_key(user_id, scene_id, phase_number, header)
check_idempotency(endpoint, key, payload)   → cached result if exists
mark_idempotency_completed(endpoint, key, result)
mark_idempotency_failed(endpoint, key, error)
```

### API Rate Limiting

Enforced via `slowapi` on all endpoints. Limits are configurable in `app/main.py` via standard slowapi decorators.

---

## Appendix: Key File Map

```
backend/services/production/
├── app/
│   ├── main.py                          FastAPI app + health checks
│   ├── config.py                        MongoDB/S3 singletons + env vars
│   ├── celery_app.py                    Celery + SQS configuration
│   ├── api/v1/endpoints/
│   │   ├── projects.py                  Project CRUD
│   │   ├── phase1.py                    Phase 1 routes
│   │   ├── phase2.py                    Phase 2 routes
│   │   ├── phase3.py                    Phase 3 routes
│   │   ├── movies.py                    Movie management
│   │   └── master.py                    End-to-end pipeline
│   ├── services/
│   │   ├── phase_1_agents/
│   │   │   ├── langgraph_workflow.py    Phase 1 LangGraph orchestrator
│   │   │   ├── workflow_state.py        Phase1State TypedDict
│   │   │   └── agent_{1..8}_*.py        Individual agent classes
│   │   ├── phase_2_agents/
│   │   │   ├── langgraph_workflow.py    Phase 2 LangGraph orchestrator
│   │   │   ├── workflow_state.py        Phase2State TypedDict
│   │   │   ├── helpers/asset_library.py Phase 1 → Phase 2 asset bridge
│   │   │   └── {agent directories}/    Individual agent classes
│   │   ├── phase_3_agents/
│   │   │   ├── langgraph_workflow.py    Phase 3 LangGraph orchestrator
│   │   │   ├── workflow_state.py        Phase3State TypedDict
│   │   │   ├── video_prompt_A/          Text-to-video prompt agent
│   │   │   ├── video_prompt_B/          Image-to-video prompt agent
│   │   │   └── video_generation/        Veo API + review agents
│   │   ├── project_service.py           Project CRUD
│   │   ├── pipeline_service.py          Job tracking
│   │   ├── shots_service.py             Shots collection operations
│   │   ├── movie_service.py             Movie management
│   │   └── assets_collection_service.py Shared asset library
│   ├── tasks/
│   │   ├── phase1_tasks.py              Phase 1 Celery tasks
│   │   ├── phase2_tasks.py              Phase 2 Celery tasks
│   │   ├── phase3_tasks.py              Phase 3 Celery tasks
│   │   └── master_tasks.py             Full pipeline orchestration tasks
│   ├── models/
│   │   ├── requests.py                  Pydantic request schemas
│   │   ├── responses.py                 Pydantic response schemas
│   │   └── mongodb/
│   │       ├── projects.py              projects collection schema
│   │       ├── movies.py                movies collection schema
│   │       ├── shots.py                 shots collection schema
│   │       ├── pipelines.py             pipelines collection schema
│   │       ├── assets.py                assets collection schema
│   │       └── assets_collections.py    assets_collections schema
│   ├── core/
│   │   ├── quota.py                     Redis-backed quota manager
│   │   └── idempotency.py               Idempotency key service
│   └── utils/
│       ├── s3_helpers.py                S3 upload/download
│       ├── csv_parser.py                Movie CSV parsing
│       └── name_normalization.py        Entity name normalisation
├── start.sh                             Service startup script
├── celery_worker.sh                     Worker startup helper
└── cleanup_workers.sh                   Process cleanup utility
```
