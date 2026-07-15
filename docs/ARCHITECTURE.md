# Zeroshot Studio — Architecture Diagrams

---

## 1. System Overview

End-to-end infrastructure: client request → API → async task queue → AI pipeline workers → storage + AI models.

```mermaid
graph TD
    Client["Client / Frontend"]

    subgraph API_Layer["FastAPI Service — port 8000 (app/main.py)"]
        Router["API Router"]
        EP_Projects["GET/POST /api/v1/projects"]
        EP_P1["POST /api/v1/phase1"]
        EP_P2["POST /api/v1/phase2"]
        EP_P3["POST /api/v1/phase3"]
        EP_Movies["POST /api/v1/movies"]
        EP_Master["POST /api/v1/master/run-pipeline"]
        EP_Health["GET /health"]
    end

    subgraph Queue_Layer["Async Task Layer (app/celery_app.py)"]
        SQS[("AWS SQS\nproduction_WORKFLOW_QUEUE")]
        CeleryWorkers["Celery Workers\n3 retries · 2hr timeout"]
    end

    subgraph Pipeline_Workers["AI Pipeline Workers (LangGraph)"]
        PW1["Phase 1 Worker\nAsset Generation — 8 Agents"]
        PW2["Phase 2 Worker\nShot Images — 18 Agents per Scene"]
        PW3["Phase 3 Worker\nVideo Generation — 3 Agents per Shot"]
    end

    subgraph Storage_Layer["Data & Storage"]
        Mongo[("MongoDB\nproduction_projects\nshots · movies\npipelines · assets\nassets_collections")]
        S3[("AWS S3\nImages & Videos")]
        Redis[("Redis DB 1\nQuota · Idempotency")]
    end

    subgraph AI_Models["AI Model Providers — Google Cloud"]
        Gemini["Gemini Pro / Flash\nText Reasoning + Vision Review"]
        Imagen["Google Imagen 4.0\nImage Generation"]
        Veo["Gemini Veo 3.1\nText-to-Video & Image-to-Video"]
        SeeDream["SeeDream 4 Edit\nImage Editing & Refinement"]
    end

    Client -->|"HTTP REST"| Router
    Router --> EP_Projects & EP_P1 & EP_P2 & EP_P3 & EP_Movies & EP_Master & EP_Health
    EP_P1 & EP_P2 & EP_P3 & EP_Master -->|"enqueue Celery task"| SQS
    SQS --> CeleryWorkers
    CeleryWorkers --> PW1 & PW2 & PW3

    PW1 & PW2 & PW3 -->|"read / write state"| Mongo
    PW1 & PW2 & PW3 -->|"upload assets"| S3
    Router -->|"quota + dedup check"| Redis

    PW1 -->|"structured prompts"| Gemini
    PW1 -->|"asset image gen"| Imagen
    PW1 -->|"image editing"| SeeDream

    PW2 -->|"shot prompts + review"| Gemini
    PW2 -->|"shot image gen"| Imagen
    PW2 -->|"image editing"| SeeDream

    PW3 -->|"video prompt + AI review"| Gemini
    PW3 -->|"video synthesis"| Veo
```

---

## 2. Pre-Production Pipeline (Strategy → Script)

Three sequential phases transforming a brand brief into a production-ready shotlist.

```mermaid
graph LR
    Brief["Brand Brief\nCompany + Product Info"]

    subgraph PreProd1["Pre-Production Phase 1 — Strategic Foundation"]
        direction TB
        PP1_1["Company Research"]
        PP1_2["Visual Context Extraction"]
        PP1_3["Brand Adjective Agent"]
        PP1_4["Audience Persona"]
        PP1_5["Competitive Landscape"]
        PP1_6["Central Human Truth"]
        PP1_7["Value Prop & Offer"]
        PP1_8["Truest Thing"]
        PP1_9["Conflict Identification"]
        PP1_10["Insight Validation"]
        PP1_11["Truth Conflict Platform"]
        PP1_12["Strategy Models"]
        PP1_13["Positioning Alignment"]
        PP1_1 --> PP1_2 --> PP1_3 --> PP1_4 --> PP1_5 --> PP1_6
        PP1_6 --> PP1_7 --> PP1_8 --> PP1_9 --> PP1_10 --> PP1_11 --> PP1_12 --> PP1_13
    end

    subgraph PreProd2["Pre-Production Phase 2 — Creative Development"]
        direction TB
        PP2_1["Creative Constraint Manager"]
        PP2_2["Constraint Priority Resolver"]
        PP2_3["Idea Core Preservation"]
        PP2_4["Brand Guideline Alignment"]
        PP2_5["Video Type Selection"]
        PP2_6["Duration Structuring"]
        PP2_7["Scene Deconstruction"]
        PP2_8["Scene Role Enumeration"]
        PP2_9["Scene Role Selector"]
        PP2_10["Temporal Placement Solver"]
        PP2_11["Scene Integration Plan"]
        PP2_12["Narrative Skeleton Generator"]
        PP2_13["Narrative Skeleton Planner"]
        PP2_14["Narrative Archetype Selector"]
        PP2_15["Platform Behavior Optimizer"]
        PP2_16["Pattern Interrupt Generator"]
        PP2_17["Viral Mechanics Agent"]
        PP2_18["Mental Model Transformer"]
        PP2_19["Intergalactic Thinking"]
        PP2_20["Concept Generator"]
        PP2_21["Offer Narrative Integrator"]
        PP2_22["Concept Categorization"]
        PP2_23["Concept Diversity Controller"]
        PP2_24["Interest Filter"]
        PP2_25["Concept Kill Switch"]
        PP2_1 --> PP2_2 --> PP2_3 --> PP2_4 --> PP2_5 --> PP2_6 --> PP2_7
        PP2_7 --> PP2_8 --> PP2_9 --> PP2_10 --> PP2_11 --> PP2_12 --> PP2_13
        PP2_13 --> PP2_14 --> PP2_15 --> PP2_16 --> PP2_17 --> PP2_18 --> PP2_19
        PP2_19 --> PP2_20 --> PP2_21 --> PP2_22 --> PP2_23 --> PP2_24 --> PP2_25
    end

    subgraph PreProd3["Pre-Production Phase 3 — Audio-Visual Scripting"]
        direction TB
        PP3_1["Beat to Timeline Mapper"]
        PP3_2["Offer Integration Planner"]
        PP3_3["Visual Sequencing"]
        PP3_4["AV Separation"]
        PP3_5["Voiceover Writer"]
        PP3_6["Dialogue Agent"]
        PP3_7["Audio Design"]
        PP3_8["Rhythm & Pacing Regulator"]
        PP3_9["Loop Optimization"]
        PP3_10["Constraint Compliance QA"]
        PP3_11["Shot Level Script Formatter"]
        PP3_12["Final Shotlist Agent"]
        PP3_1 --> PP3_2 --> PP3_3 --> PP3_4 --> PP3_5 --> PP3_6
        PP3_6 --> PP3_7 --> PP3_8 --> PP3_9 --> PP3_10 --> PP3_11 --> PP3_12
    end

    ShotlistOutput["Production-Ready Shotlist\n+ Script + Audio Specs"]

    Brief --> PP1_1
    PP1_13 -->|"strategy + insights"| PP2_1
    PP2_25 -->|"approved concepts"| PP3_1
    PP3_12 --> ShotlistOutput
    ShotlistOutput -->|"input to Production Phase 1"| ProdP1["Production Pipeline"]
```

---

## 3. Production Phase 1 — Asset Generation (Agents 1–8)

Takes the script and generates a complete visual asset library (characters, locations, props) with AI-reviewed images and camera-angle variations.

```mermaid
graph TD
    Script["Script + Shotlist CSV\n+ Visual Style + Product Image"]

    A1["Agent 1 — Asset Extractor\nExtract characters, locations, props\nGemini structured output"]
    A2["Agent 2 — Asset Reviewer\nValidate & enhance extracted assets\nCSV-filter phantom entities"]
    A3["Agent 3 — Prompt Generator\nWrite visual image prompts\nper asset × visual style"]
    A4["Agent 4 — Prompt Optimizer\nRefine prompts for Imagen quality"]
    A5["Agent 5 — Image Generator\nGoogle Imagen 4.0\nSkip product (use uploaded image)"]
    A6{"Agent 6 — Image Reviewer\nGemini Vision multimodal\nDecision per image"}
    A7["Agent 7 — Image Editor\nSeeDream 4 Edit API\nMax 3 edit iterations"]
    A8["Agent 8 — Variation Generator\nCamera angle variations\nback · close-up · profile · wide\nCharacters only"]

    HC1{{"Human Checkpoint\nApprove Asset Library"}}
    Approve["Approved\nAssets saved to MongoDB\nassets_collections / agent_outputs"]
    Reject["Reject → Regenerate\nrestart from Agent 5"]
    RetryAsset["Retry Single Asset\n/phase1/retry-asset endpoint"]

    Script --> A1
    A1 --> A2 --> A3 --> A4 --> A5 --> A6

    A6 -->|"all approved"| A8
    A6 -->|"needs edit"| A7
    A6 -->|"regenerate"| A5

    A7 -->|"re-review (max 3 loops)"| A6
    A8 --> HC1

    HC1 -->|"approve"| Approve
    HC1 -->|"reject"| Reject
    HC1 -->|"retry-asset"| RetryAsset
    RetryAsset --> A5
```

---

## 4. Production Phase 2 — Shot Image Generation (Agents 1–18)

Runs **per scene**. Uses Phase 1 asset library to generate composition-aware images for every shot. Three human checkpoints gate strategy, prompts, and final images.

```mermaid
graph TD
    ShotList["Shot List\nshow_id + episode_number\n+ Phase 1 Asset Library"]

    A2_1["Agent 1 — Shot Strategy\nAnalyse each shot\nAssign generation_strategy:\ngenerate_new / last_frame_seed / multi_shot"]

    HC2_1{{"Human Checkpoint 1\nApprove Shot Strategies"}}

    A2_2["Agent 2 — Image Prompt Generator\nCinematic prompts per shot\nusing AssetLibrary (helpers/asset_library.py)"]
    A2_3["Agent 3 — Prompt Reviewer\nQA on generated prompts"]
    A2_12["Agent 12 — Shot Design\nAsset selection + composition strategy\nPromptFeasibilityChecker"]
    A2_13["Agent 13 — Prompt Modifier\nAdapt prompts to available assets"]

    HC2_2{{"Human Checkpoint 2\nApprove Prompts"}}

    A2_14["Agent 14 — Imagen Generator\nGoogle Imagen 4.0\nVersion-tracked: v0, v1, v2…"]
    A2_15{"Agent 15 — Image Reviewer\nGemini Vision per shot\nDecision per image"}
    A2_15A["Agent 15A — Prompt Regeneration\nRewrite prompts from review feedback"]
    A2_7["Agent 7 — Shot Editor\nSeeDream 4 Edit\nMax 3 edit iterations"]

    ProductCheck{"product_image_url present\nAND shot has product?"}
    A2_16["Agent 16 — Product Reviewer\nGemini Vision\nFidelity check on product shots"]
    A2_17["Agent 17 — Product Prompt Gen\nFix product appearance issues"]
    A2_18["Agent 18 — Product Editor\nFinal product shot touch-ups\nMax 3 iterations"]

    HC2_3{{"Human Checkpoint 3\nFinal Image Approval"}}
    FinalImages["Shot Images Saved\nshots collection\nversioned by shot_id"]

    ShotList --> A2_1
    A2_1 --> HC2_1
    HC2_1 -->|"approved"| A2_2

    A2_2 --> A2_3 --> A2_12 --> A2_13
    A2_13 --> HC2_2
    HC2_2 -->|"approved"| A2_14

    A2_14 --> A2_15
    A2_15 -->|"approved"| ProductCheck
    A2_15 -->|"regenerate (max 3)"| A2_15A
    A2_15 -->|"edit needed"| A2_7
    A2_15A -->|"new prompt"| A2_14
    A2_7 -->|"re-review (max 3 loops)"| A2_15

    ProductCheck -->|"yes — product shots"| A2_16
    ProductCheck -->|"no"| HC2_3
    A2_16 -->|"needs fix"| A2_17
    A2_16 -->|"approved"| HC2_3
    A2_17 --> A2_18
    A2_18 -->|"re-review (max 3 loops)"| A2_16

    HC2_3 -->|"final approved"| FinalImages
```

---

## 5. Production Phase 3 — Video Generation (Agents 17–19)

Runs **per shot**. Reads shot images from MongoDB, selects the right prompt strategy, calls Gemini Veo 3.1, applies AI review, then human approval.

```mermaid
graph TD
    ShotDB["Shot Data from MongoDB\nshot_id + generation_strategy\n+ reference image S3 URL"]

    StrategyCheck{"Check generation_strategy"}

    PromptA["Agent 17A — Video Prompt A\nDetailed cinematic description\nfor text-to-video\nvideo_prompt_A/agent_video_generation.py"]
    PromptB["Agent 17B — Video Prompt B\nConcise prompt (max 30 words)\nfor image-to-video consistency\nvideo_prompt_B/video_prompt_B.py"]

    VeoAPI["Gemini Veo 3.1\nvideo_generation_api_agent.py"]
    TextToVideo["Text-to-Video\ngenerate_new strategy"]
    ImgToVideoFirst["Image-to-Video\nmulti_shot — first frame reference"]
    ImgToVideoLast["Image-to-Video\nlast_frame_seed — last frame seed"]

    AIReview{"Agent 19 — AI Video Review\nGemini Vision multimodal\nvideo_review_agent.py"}

    Refine["Refine Prompt\nsuggest_prompt from review\nMax 3 attempts"]
    Retry["Regenerate\ndifferent parameters\nMax 3 attempts"]

    HC3{{"Human Checkpoint\nVideo Approval"}}

    ChangesLoop["Regenerate with\nhuman feedback prompt\nMax 3 attempts"]
    Complete["Video Saved\nS3 URL + MongoDB\nshots.video versioned"]

    ShotDB --> StrategyCheck

    StrategyCheck -->|"generate_new\nor last_frame_seed"| PromptA
    StrategyCheck -->|"multi_shot"| PromptB

    PromptA -->|"text prompt"| VeoAPI
    PromptB -->|"short prompt + ref image"| VeoAPI

    VeoAPI --> TextToVideo
    VeoAPI --> ImgToVideoFirst
    VeoAPI --> ImgToVideoLast

    TextToVideo & ImgToVideoFirst & ImgToVideoLast -->|"poll completion"| AIReview

    AIReview -->|"approved"| HC3
    AIReview -->|"refine_prompt"| Refine
    AIReview -->|"regenerate"| Retry
    Refine -->|"new prompt → Veo"| VeoAPI
    Retry -->|"retry → Veo"| VeoAPI

    HC3 -->|"approved"| Complete
    HC3 -->|"needs_changes"| ChangesLoop
    ChangesLoop -->|"human prompt → Veo"| VeoAPI
```

---

## 6. Full Production Data Flow

How data passes between phases end-to-end.

```mermaid
graph LR
    subgraph Input["Inputs"]
        Script2["Script Text"]
        ShotListCSV["Shotlist CSV"]
        ProductImg["Product Image"]
        VisualStyle["Visual Style"]
    end

    subgraph P1_Out["Phase 1 Outputs — MongoDB"]
        Characters["Characters\nImages + Variations\n(6 camera angles each)"]
        Locations["Locations\nImages"]
        Props["Props\nImages"]
    end

    subgraph P2_Out["Phase 2 Outputs — MongoDB shots collection"]
        ShotImages["Shot Images\nv0 · v1 · v2 (versioned)\ngeneration_strategy per shot"]
        ShotPrompts["Shot Prompts\nAgent 13 modified"]
    end

    subgraph P3_Out["Phase 3 Outputs — MongoDB + S3"]
        Videos["Videos per Shot\nS3 URL + version history"]
        FinalVideo["Final Assembled Video"]
    end

    AssetLibrary["AssetLibrary\nhelpers/asset_library.py\nBridges Phase 1 → Phase 2"]

    Script2 & ShotListCSV & ProductImg & VisualStyle -->|"Phase 1 input"| Characters & Locations & Props
    Characters & Locations & Props -->|"loaded by"| AssetLibrary
    AssetLibrary -->|"asset selection\n+ composition"| ShotImages
    ShotListCSV -->|"shot descriptions"| ShotImages
    ShotImages -->|"reference frame"| Videos
    ShotPrompts -->|"video prompt input"| Videos
    Videos --> FinalVideo
```
