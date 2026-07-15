# Product Image Pipeline — Implementation & Consistency Fix Guide

## What was built

The pipeline previously fetched product images from a **legacy v1 MongoDB database** using a `v1_project_id` parameter that had to be passed manually. The production service had no ownership over that data. This was replaced with a self-contained product image upload flow: the image is uploaded at project creation time, stored in S3, the URL is saved in production MongoDB, and both Phase 1 and Phase 2 read from there automatically.

---

## Changes Made

### 1. MongoDB Schema — `app/models/mongodb/projects.py`

Added `product_image_s3_url` to `PROJECTS_SCHEMA`. Because the schema uses `additionalProperties: True`, this is additive — no migration needed.

```python
"product_image_s3_url": {
    "bsonType": ["string", "null"],
    "description": "S3 URL of the uploaded product image for product shots"
},
```

---

### 2. Project Service — `app/services/project_service.py`

Added `update_product_image_url()` alongside `update_s3_urls`:

```python
def update_product_image_url(self, project_id: str, s3_url: str) -> bool:
    client, projects_col = get_projects_collection()
    project_obj_id = validate_object_id(project_id)
    result = projects_col.update_one(
        {"_id": project_obj_id},
        {"$set": {"product_image_s3_url": s3_url, "updated_at": datetime.utcnow()}}
    )
    ...
```

---

### 3. Projects Endpoint — `app/api/v1/endpoints/projects.py`

`POST /projects/{id}/upload-files` now accepts an optional `product_image_file` (PNG/JPG/JPEG).

- Validated extension, read bytes, wrote temp file
- S3 key: `projects/{project_id}/product_image{ext}`
- Called `project_service.update_product_image_url(project_id, product_image_url)` to store in MongoDB
- Removed `v1_project_id` Form parameter

---

### 4. Movies Endpoint — `app/api/v1/endpoints/movies.py`

`POST /movies/create` now accepts an optional `product_image_file`.

- Uploaded once at movie level: S3 key `movies/{movie_id}/product_image{ext}`
- Stored the same URL in every scene project document via `project_service.update_product_image_url(pid, product_image_url)`
- Removed `v1_project_id` Form parameter and from all downstream calls

---

### 5. Master Endpoint — `app/api/v1/endpoints/master.py`

`POST /master/run-pipeline` now accepts an optional `product_image_file`.

Because `UploadFile` cannot be serialized to Celery/SQS, the upload happens in the endpoint handler **before** dispatching the task:

- S3 key: `master/{master_job_id}/product_image{ext}`
- `product_image_s3_url` is passed as a serializable string kwarg to `run_master_pipeline_task.apply_async`

---

### 6. Master Task — `app/tasks/master_tasks.py`

Added `product_image_s3_url: Optional[str] = None` parameter. After each scene project is created inside the task, calls `proj_svc.update_product_image_url(project_id, product_image_s3_url)` if the URL is set. This ensures all scene projects have the image URL before their Phase 1 run starts.

---

### 7. Phase 2 Endpoint — `app/api/v1/endpoints/phase2.py`

Removed `v1_project_id` from the `Phase2StartRequest` Pydantic model and from the Celery kwargs dispatch. The field no longer appears in Swagger.

---

### 8. Phase 1 Workflow State — `app/services/phase_1_agents/workflow_state.py`

Added `product_image_s3_url: Optional[str]` to `Phase1State`.

---

### 9. Agent 1 Prompt — `app/services/phase_1_agents/prompts.py`

Updated `Agent1Prompts.asset_extraction` to accept `product_image_available: bool = False`. When `True`, this block is appended to the prompt:

```
PRODUCT IMAGE RULE — READ CAREFULLY:
A real product image has been uploaded for this project. The script features an advertised product.
When extracting props, identify which single prop represents the main advertised product being featured.
Mark that prop with "is_product": true in your JSON output.
Only ONE prop should have "is_product": true.
```

---

### 10. Agent 1 — `app/services/phase_1_agents/agent_1_asset_generator.py`

Added `product_image_available: bool = False` to `__init__`. Passed to `_create_asset_extraction_prompt`.

---

### 11. Phase 1 Workflow — `app/services/phase_1_agents/langgraph_workflow.py`

#### Initialization (replaced v1 DB fetch)

Reads `product_image_s3_url` from the production project document via `project_service.get_project(project_id)`. Added to `initial_state`.

#### Agent 1 node — post-processing

After Agent 1 returns extracted assets:

1. If `product_image_s3_url` is set and Agent 1 flagged a prop with `is_product: True` → attach `pre_generated_image_url = product_image_s3_url` to that prop.
2. If Agent 1 didn't flag any prop → inject a fallback PRODUCT prop with `is_product: True` and `pre_generated_image_url = product_image_s3_url`.
3. Legacy path: if no uploaded image but `state["product_prop"]` exists → append it (backward compat).

#### Agent 5 node — generalized exclusion/injection

Instead of excluding by prop ID, now excludes any prop with `is_product: True` from Imagen generation. Instead of reading from `state["product_prop"]`, injects uploaded image for every prop that has both `is_product: True` and `pre_generated_image_url`.

---

### 12. Agent 6 Image Reviewer — `app/services/phase_1_agents/agent_6_image_reviewer.py`

For props with `is_product: True`:
- If the LLM returns `"regenerate"`, it is forcibly overridden to `"approved"`.
- Review note is set: `"Product image auto-approved — regeneration not permitted for uploaded product images."`

The product's shape, size, text, and logo are never evaluated for replacement.

---

### 13. Agent 7 Image Editor — `app/services/phase_1_agents/agent_7_image_editor.py`

When generating the edit prompt for a prop with `is_product: True`, this prefix is prepended:

```
CRITICAL — PRODUCT FIDELITY: Under NO circumstances alter the product's
shape, size, proportions, text, logo, label, color, or branding.
Permitted adjustments ONLY: background, shadows, reflections, lighting intensity.
The product itself must remain pixel-perfect.
```

---

### 14. Phase 2 Workflow — `app/services/phase_2_agents/langgraph_workflow.py`

Replaced the v1 DB fetch block with a production MongoDB lookup:

```python
from backend.services.production.app.services.project_service import ProjectService as _PS
_proj_doc = _PS().get_project(project_id)
product_image_url = _proj_doc.get("product_image_s3_url") if _proj_doc else None
```

`product_image_url` is stored in Phase 2 state and injected into `corrected_assets` for every shot where `product_present=True` (in Agent 13 and Agent 14 nodes). `v1_project_id` is kept in the state dict for backward compat but no longer used for product image fetching.

---

## Why Phase 2 Still Generates a Wrong Product

Despite the correct URL being fetched and the reference image being passed to Gemini, the generated shot may still show a wrong-looking product (e.g., a plain white cylinder instead of the actual branded silver tin). There are four compounding root causes.

### Root Cause 1 — Presigned URL expiry (most common silent failure)

`upload_file_wrapper` generates a presigned URL with a default TTL (24 hours by default in most S3 SDK configs). If Phase 2 runs after that window, `_fetch_image_from_url` silently returns `None`. The code path then falls through to text-only generation — Gemini sees no reference image and invents a product from the prompt text alone.

**Evidence**: The logger will show `❌ Failed to fetch ... - Status: 403` or `Status: 400`. If no error is logged at all (URL returned `None` before the request), the corrected_assets injection didn't work.

**Where it happens**: `imagen_generator_agent.py:_fetch_image_from_url` — returns `None` on non-200, silently skips.

---

### Root Cause 2 — Weak reference label for product assets

When Gemini is called, each reference image gets a text label immediately before it in the contents array. The current label for product assets is:

```
Reference image for the Product (PRODUCT):
```

This is the same generic label used for characters and locations. Gemini treats it as a stylistic reference — "something like this" — not as a blueprint to copy exactly. For characters, this is fine. For a product whose exact appearance (tin shape, label text, color, lid design) must be reproduced, this instruction is far too weak.

**Where it happens**: `imagen_generator_agent.py:_generate_image_gemini_flash` lines 324–335 (also `_generate_image_gemini_pro` lines 441–452).

---

### Root Cause 3 — No product fidelity instruction in the final Gemini prompt

The final instruction sent to Gemini is:

```
Generate a realistic cinematic image combining these reference images above. {prompt}
```

"Combining" tells Gemini to blend and interpret. There is no instruction that says "the product must look exactly like the reference — preserve its shape, text, label, and proportions unchanged." Gemini's image model is a generative model that will creatively interpret unless explicitly constrained.

**Where it happens**: `imagen_generator_agent.py:_generate_image_gemini_flash` line 349 (and `_generate_image_gemini_pro` line 465).

---

### Root Cause 4 — Prompt generation (Agent 2/9) suppresses product description

The Phase 2 prompt generation agent is instructed: *"DO NOT describe the product's appearance — the model sees it directly."* This means the corrected prompt passed to Gemini contains no product appearance description at all. If the reference image fails to load (Root Cause 1), Gemini has neither a reference image nor a textual description of the product — it invents something from the product name alone.

**Where it happens**: Phase 2 `image_prompt_generator_agent.py` (system prompt for Agent 2/9).

---

## How to Fix Product Consistency in Phase 2

### Fix 1 — Store product image with a permanent URL (critical, already applied)

The file IS uploaded to S3 permanently — it never disappears. What expires is the **presigned URL** (the temporary signed access token) stored in MongoDB. `upload_file_wrapper` defaults to `use_presigned_url=True` with a 24-hour TTL. After that window, `_fetch_image_from_url` gets a 403 and silently returns `None`.

**Fix applied**: All three product image upload calls (`projects.py`, `movies.py`, `master.py`) now pass `use_presigned_url=False`, which causes `infrastructure/s3/upload.py` to return the direct permanent S3 URL (`https://{bucket}.s3.{region}.amazonaws.com/{key}`) instead of a presigned token.

```python
product_image_url = upload_file_wrapper(
    tmp_path,
    s3_key=s3_key,
    content_type=...,
    use_presigned_url=False,   # ← permanent URL, no expiry
)
```

**Requirement**: The S3 bucket must allow public GET access for this to work. If the bucket has public access blocked, use `presigned_expiration=31536000` (1 year) as a fallback, but a public bucket is cleaner.

**Verification**: After uploading a product image, check `product_image_s3_url` in MongoDB — it should be a plain `https://` URL (no `X-Amz-Expires` query parameter). Fetching it should return HTTP 200 at any time.

---

### Fix 2 — Strengthen the product reference label in Agent 14

**File**: `app/services/phase_2_agents/imagen_generator_agent.py`

In `_generate_image_gemini_flash` and `_generate_image_gemini_pro`, inside the loop that builds `contents`, change the label for product-type assets:

```python
# Before (line ~331):
label = f"Reference image for the {asset_type} ({asset_name}):"

# After:
if asset.get('type') == 'product':
    label = (
        f"PRODUCT REFERENCE — REPRODUCE THIS EXACTLY:\n"
        f"The product in the generated image MUST have the identical shape, dimensions, label text, "
        f"logo, colors, lid design, and branding as shown in this reference image. "
        f"Do NOT substitute, simplify, or reinterpret the product's appearance:"
    )
else:
    label = f"Reference image for the {asset_type} ({asset_name}):"
```

Apply this change in **both** `_generate_image_gemini_flash` and `_generate_image_gemini_pro`.

---

### Fix 3 — Add product fidelity constraint to the final generation instruction

**File**: `app/services/phase_2_agents/imagen_generator_agent.py`

After building `final_instruction` (line ~348 in Flash, ~464 in Pro), check whether a product asset is present and append a constraint:

```python
has_product_asset = assets_metadata and any(
    a.get('type') == 'product' for a in assets_metadata
)

if has_product_asset:
    product_fidelity = (
        "\n\nCRITICAL PRODUCT FIDELITY RULE: One of the reference images above is an actual product. "
        "In the generated image, this product must appear with EXACTLY the same: "
        "shape, size, label text, logo, colors, lid/cap design, and overall form. "
        "Do NOT simplify, stylize, or substitute the product's appearance. "
        "Reproduce it faithfully as if photographed."
    )
    final_instruction = final_instruction + product_fidelity
```

Apply to both `_generate_image_gemini_flash` and `_generate_image_gemini_pro`.

---

### Fix 4 — Add product appearance description as fallback in prompt generation

**File**: `app/services/phase_2_agents/image_prompt_generator_agent.py` (Agent 2/9)

The instruction "DO NOT describe the product's appearance" is correct when the reference image is reliably available. But as a safety net when the image fails to load, the prompt should contain a minimal appearance description derived from the project's product description.

Change the agent instruction to:

```
When generating the shot prompt for a product shot:
1. Do NOT describe the product's appearance if a reference image is provided (Gemini sees it directly).
2. However, always include a one-sentence product description in parentheses as a fallback:
   e.g., "(Product: square silver tin with white rounded lid, red strawberry pomegranate lip balm visible, branded label)"
   This ensures correct appearance even if the reference image cannot be loaded.
```

This fallback description should be sourced from `scene_description` or extracted from the Phase 1 prop description stored in MongoDB.

---

### Fix 5 — Detect and alert when product image fetch fails (observability)

**File**: `app/services/phase_2_agents/imagen_generator_agent.py`

In `generate_images_for_shots`, after the asset fetching loop, add a check:

```python
# After the corrected_assets fetching loop
expected_product = any(a.get('type') == 'product' for a in corrected_assets)
fetched_product = any(a.get('type') == 'product' for a in assets_metadata)

if expected_product and not fetched_product:
    logger.error(
        f"❌ PRODUCT IMAGE FETCH FAILED for {shot_id} — "
        f"product reference image could not be loaded. "
        f"The generated image will NOT have the correct product appearance. "
        f"Check that product_image_s3_url is not expired."
    )
```

This makes the failure loud instead of silent, so logs clearly indicate when the product reference image was missing during generation.

---

## Summary of Priority

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 1 (critical) | Fix URL expiry (Fix 1) | Eliminates silent fallback to text-only generation | Low — change one parameter |
| 2 (high) | Stronger product label (Fix 2) | Tells Gemini to copy instead of interpret | Low — 5-line change |
| 3 (high) | Product fidelity in final instruction (Fix 3) | Adds explicit constraint alongside reference | Low — 10-line change |
| 4 (medium) | Fallback product description in prompts (Fix 4) | Resilience when image fetch fails | Medium — update prompt agent |
| 5 (low) | Logging/alerting on fetch failure (Fix 5) | Observability only, no quality improvement | Low |

Fixes 1, 2, and 3 together will eliminate the majority of product inconsistency cases. Fix 1 alone will fix cases where the URL has expired. Fixes 2 and 3 together will fix cases where the image loads correctly but Gemini still reinterprets the product creatively.
