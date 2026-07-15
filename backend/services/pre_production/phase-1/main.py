import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

# Add phase-1 sub-directories to the import path so that agent modules
# (company_research, visual_context_extraction, brand_adjective, …) can be
# imported by name regardless of how uvicorn resolves the package path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "input"))
sys.path.insert(0, os.path.join(_HERE, "brand_adjective"))
sys.path.insert(0, os.path.join(_HERE, "audience_persona"))
sys.path.insert(0, os.path.join(_HERE, "competitive_landscape"))
sys.path.insert(0, os.path.join(_HERE, "central_human_truth"))
sys.path.insert(0, os.path.join(_HERE, "value_prop_and_offer"))
sys.path.insert(0, os.path.join(_HERE, "truest_thing"))
sys.path.insert(0, os.path.join(_HERE, "conflict_identification"))
sys.path.insert(0, os.path.join(_HERE, "truth_conflict_platform"))
sys.path.insert(0, os.path.join(_HERE, "strategy_models"))
sys.path.insert(0, os.path.join(_HERE, "positioning_alignment"))

# Add phase 2 directory
PHASE_2_DIR = os.path.join(os.path.dirname(_HERE), "phase-2")
sys.path.insert(0, PHASE_2_DIR)

# Add phase 3 directory
PHASE_3_DIR = os.path.join(os.path.dirname(_HERE), "phase-3")
sys.path.insert(0, PHASE_3_DIR)

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import BackgroundTasks, FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel
from company_research import run_company_research_extraction
from visual_context_extraction import run_visual_context_extraction
from brand_adjective import run_brand_adjective_agent
from audience_persona import run_audience_persona_agent
from competitive_landscape import run_competitive_landscape_agent
from central_human_truth import run_central_human_truth_agent
from value_prop_and_offer import run_value_prop_and_offer_agent
from truest_thing import run_truest_thing_agent
from conflict_identification import run_conflict_identification_agent
from insight_validation import run_insight_validation_agent
from truth_conflict_platform import run_truth_conflict_platform_agent
from strategy_models import run_strategy_models_agent
from positioning_alignment import run_positioning_alignment_agent
# Load Phase 1's orchestrator by explicit file path to avoid collision with
# phase-3/orchestrator.py, which ends up earlier in sys.path.
import importlib.util as _ilu
_p1_orch_spec = _ilu.spec_from_file_location(
    "phase1_orchestrator", os.path.join(_HERE, "orchestrator.py")
)
_p1_orch_mod = _ilu.module_from_spec(_p1_orch_spec)
_p1_orch_spec.loader.exec_module(_p1_orch_mod)
run_phase_1_pipeline = _p1_orch_mod.run_phase_1_pipeline

# Import Phase 2 Router
try:
    from main import phase2_router  # Ensure this matches the module and router name in phase-2/main.py
except ImportError:
    # If starting via 'uvicorn main:app' in phase-1, sys.path hack enables:
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase2_main", os.path.join(PHASE_2_DIR, "main.py"))
    phase2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase2_module)
    phase2_router = phase2_module.phase2_router

# Import Phase 3 Router
try:
    from main import phase3_router  # Ensure this matches the module and router name in phase-3/main.py
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase3_main", os.path.join(PHASE_3_DIR, "main.py"))
    phase3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase3_module)
    phase3_router = phase3_module.phase3_router

# Load pipeline runner functions by absolute path to avoid orchestrator name conflicts
# (both phase-2 and phase-3 have an orchestrator.py, so bare imports are ambiguous)
import importlib.util as _importlib_util

_p2_orch_spec = _importlib_util.spec_from_file_location(
    "phase2_orchestrator", os.path.join(PHASE_2_DIR, "orchestrator.py")
)
_p2_orch_mod = _importlib_util.module_from_spec(_p2_orch_spec)
_p2_orch_spec.loader.exec_module(_p2_orch_mod)
run_phase_2_pipeline = _p2_orch_mod.run_phase_2_pipeline

_p3_orch_spec = _importlib_util.spec_from_file_location(
    "phase3_orchestrator", os.path.join(PHASE_3_DIR, "orchestrator.py")
)
_p3_orch_mod = _importlib_util.module_from_spec(_p3_orch_spec)
_p3_orch_spec.loader.exec_module(_p3_orch_mod)
run_phase_3_pipeline = _p3_orch_mod.run_phase_3_pipeline

class InsightValidationRequest(BaseModel):
    project_id: str

class InsightValidationResponse(BaseModel):
    message: str

class ConflictIdentificationRequest(BaseModel):
    project_id: str

class ConflictIdentificationResponse(BaseModel):
    message: str

class TruthConflictPlatformRequest(BaseModel):
    project_id: str

class TruthConflictPlatformResponse(BaseModel):
    message: str

class Phase1PipelineRequest(BaseModel):
    project_id: str

class PreProductionRequest(BaseModel):
    project_id: str

class StrategyModelsRequest(BaseModel):
    project_id: str

class StrategyModelsResponse(BaseModel):
    message: str

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zeroshot.input")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

MONGODB_URI: str = os.getenv("MongoDB", "")
DB_NAME: str = os.getenv("DB_NAME", "v1")
COLLECTION: str = os.getenv("COLLECTION_PROJECTS", "projects")
STRATEGY_COLLECTION: str = os.getenv("COLLECTION_STRATEGY", "strategy")  # Add this

# AWS S3
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_BUCKET: str = os.getenv("S3_BUCKET", "zeroshot-v1")
S3_REGION: str = os.getenv("S3_REGION", "eu-north-1")
S3_FOLDER: str = os.getenv("S3_FOLDER", "product images")

_s3 = boto3.client(
    "s3",
    region_name=S3_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ---------------------------------------------------------------------------
# MongoDB lifecycle
# ---------------------------------------------------------------------------
_client: AsyncIOMotorClient = None
_db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _db
    if not MONGODB_URI:
        raise RuntimeError("MongoDB URI is missing — check your .env file")
    _client = AsyncIOMotorClient(MONGODB_URI)
    _db = _client[DB_NAME]
    app.state.db = _db  # Expose DB to sub-routers cleanly
    logger.info("MongoDB connected  |  db=%s  collection=%s", DB_NAME, COLLECTION)
    yield
    _client.close()
    logger.info("MongoDB connection closed")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Zeroshot Studio — Phase 1",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Step 2 wrapper — resolves strategy_id then runs visual context extraction
# ---------------------------------------------------------------------------
async def _run_visual_context_step(project_id: str, db, s3_client) -> None:
    """Query strategy doc by project_id to get its _id, then run Step 2."""
    try:
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": project_id})
        if not strategy_doc:
            logger.warning(
                "Step 2 skipped — strategy doc not found  |  project_id=%s",
                project_id,
            )
            return
        strategy_id = str(strategy_doc["_id"])
        await run_visual_context_extraction(project_id, strategy_id, db, s3_client)
    except Exception as exc:
        logger.warning(
            "Step 2 wrapper error  |  project_id=%s  error=%s", project_id, exc,
        )

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
class GenderType(str, Enum):
    male   = "male"
    female = "female"
    other  = "other"


class AudienceGenderType(str, Enum):
    """Single-select dropdown covering every gender combination, since
    Swagger UI's multipart form widget can't do a real multi-select for
    array fields — it collapses multiselect into one comma-joined string."""
    male               = "male"
    female             = "female"
    other              = "other"
    male_female        = "male, female"
    male_other         = "male, other"
    female_other       = "female, other"
    male_female_other  = "male, female, other"


class VideoType(str, Enum):
    ugc              = "UGC"
    product_beauty   = "Product Beauty"
    flatlay          = "Flatlay"
    cgi_3d_product   = "CGI/3D Product"
    realistic        = "Realistic"
    testimonial      = "Testimonial"
    animation        = "Animated"
    narrative        = "Narrative"
    satire           = "Satire"
    superficial      = "Superficial"


@app.post(
    "/projects",
    summary="Create a new project",
    tags=["Projects"],
    status_code=201,
)
async def create_project(
    background_tasks: BackgroundTasks,
    # ── Compulsory ──────────────────────────────────────────────────────────
    company_url: str = Form(
        ...,
        description="URL of the company website",
    ),
    product_image: UploadFile = File(
        ...,
        description="Product image file (jpg / png / webp …)",
    ),
    audience_age_start: int = Form(
        ...,
        ge=0,
        description="Target audience starting age (≥ 0)",
    ),
    audience_age_end: int = Form(
        ...,
        le=120,
        description="Target audience ending age (≤ 120)",
    ),
    audience_gender: AudienceGenderType = Form(
        ...,
        description="Target audience gender",
    ),
    number_of_shots: int = Form(
        ...,
        ge=1,
        description="Number of shots in the video. Each shot = exactly 8 seconds. Total duration = number_of_shots × 8.",
    ),
    product_details: str = Form(
        ...,
        description="Details about the product",
    ),
    price_and_offer: str = Form(
        ...,
        description='Pricing and offer details — e.g. "Flat 50% off", "Trial kit ₹199"',
    ),
    # ── Optional ────────────────────────────────────────────────────────────
    video_type: Optional[VideoType] = Form(
        None,
        description="(Optional) Ad format type. Leave blank for auto-selection.",
    ),
    idea: Optional[str] = Form(
        None,
        description="(Optional) Creative idea for the campaign",
    ),
    preferred_scene: Optional[str] = Form(
        None,
        description="(Optional) Preferred scene description",
    ),
    historical_performance: Optional[str] = Form(
        None,
        description="(Optional) Previous campaign performance to avoid repeating failed angles",
    ),
    brand_guidelines: Optional[str] = Form(
        None,
        description="(Optional) Brand guidelines, banned words, visual rules — plain text or S3 PDF URL",
    ),
    sitemap_url: Optional[str] = Form(
        None,
        description="(Optional) Sitemap URL — crawler uses this directly instead of auto-discovering the sitemap",
    ),
    product_url: Optional[str] = Form(
        None,
        description="(Optional) Product webpage URL — agents use this to research and understand the product in depth",
    ),
):
    # -- Validate age range --------------------------------------------------
    if audience_age_start > audience_age_end:
        raise HTTPException(
            status_code=422,
            detail="audience_age_start must be less than or equal to audience_age_end.",
        )

    # audience_gender is a single combo value (e.g. "male, female") — split
    # it back into individual GenderType members for storage.
    gender_list = [GenderType(g.strip()) for g in audience_gender.value.split(",")]

    video_length_seconds: int = number_of_shots * 8

    logger.info("POST /projects  |  company_url=%s", company_url)

    # -- Upload product image to S3 -----------------------------------------
    ext = os.path.splitext(product_image.filename or "")[-1].lower() or ".bin"
    s3_key = f"{S3_FOLDER}/{uuid.uuid4().hex}{ext}"

    try:
        _s3.upload_fileobj(
            product_image.file,
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": product_image.content_type or "application/octet-stream"},
        )
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 upload failed  |  key=%s  error=%s", s3_key, e)
        raise HTTPException(status_code=502, detail=f"S3 upload failed: {str(e)}")

    s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    logger.info("S3 upload OK  |  key=%s", s3_key)

    # -- Build structured document ------------------------------------------
    document = {
        "company_url": company_url,
        "product_image": {
            "original_filename": product_image.filename,
            "s3_url": s3_url,
            "s3_key": s3_key,
        },
        "target_audience": {
            "age": {
                "start": audience_age_start,
                "end": audience_age_end,
            },
            "gender": [g.value for g in gender_list],
        },
        "number_of_shots": number_of_shots,
        "video_length_seconds": video_length_seconds,
        "product_details": product_details,
        "price_and_offer": price_and_offer,
        "video_type": video_type.value if video_type is not None else "TBD",
        "idea": idea,
        "preferred_scene": preferred_scene,
        "historical_performance": historical_performance,
        "brand_guidelines": brand_guidelines,
        "sitemap_url": sitemap_url,
        "product_url": product_url,
    }

    # -- Insert into MongoDB ------------------------------------------------
    result = await _db[COLLECTION].insert_one(document)
    project_id = str(result.inserted_id)
    logger.info("Project saved  |  project_id=%s", project_id)

    # -- Fire pre-processing step 1 as a background task --------------------
    background_tasks.add_task(
        run_company_research_extraction,
        project_id=project_id,
        company_url=company_url,
        db=_db,
        sitemap_url=sitemap_url,
        product_url=product_url,
    )
    logger.info("Background task queued  |  step=company_research_extraction  project_id=%s", project_id)

    # -- Fire pre-processing step 2 as a sequential background task ---------
    background_tasks.add_task(
        _run_visual_context_step,
        project_id=project_id,
        db=_db,
        s3_client=_s3,
    )
    logger.info("Background task queued  |  step=visual_context_extraction  project_id=%s", project_id)

    return {
        "message": "Project created successfully.",
        "project_id": project_id,
    }


# ---------------------------------------------------------------------------
# Brand Adjective Agent — request / response schemas
# ---------------------------------------------------------------------------

class BrandAdjectiveRequest(BaseModel):
    project_id: str


class BrandAdjectiveResponse(BaseModel):
    project_id:      str
    brand_adjective: str
    reasoning:       str


# ---------------------------------------------------------------------------
# Endpoint — POST /brand-adjective
# ---------------------------------------------------------------------------

@app.post(
    "/brand-adjective",
    response_model=BrandAdjectiveResponse,
    summary="Run the brand adjective extraction agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def brand_adjective_endpoint(body: BrandAdjectiveRequest) -> BrandAdjectiveResponse:
    """
    Triggers the brand adjective agent for an existing project.

    Reads `company_research.raw_text` and `visual_context_summary` from the
    strategy document, calls Gemini (google_search + url_context), and writes
    the result to `strategy.agents.brand_adjective` and `pipeline.agent_logs[]`.

    Errors:
    - **422** — missing / empty required fields, document not found
    - **500** — Gemini API failure, MongoDB write failure
    """
    logger.info("POST /brand-adjective  |  project_id=%s", body.project_id)

    try:
        result = await run_brand_adjective_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return BrandAdjectiveResponse(
        project_id=body.project_id,
        brand_adjective=result.brand_adjective,
        reasoning=result.reasoning,
    )


# ---------------------------------------------------------------------------
# Audience Persona Agent — request / response schemas
# ---------------------------------------------------------------------------

class AudiencePersonaRequest(BaseModel):
    project_id: str


class AudiencePersonaResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoint — POST /audience-persona
# ---------------------------------------------------------------------------

@app.post(
    "/audience-persona",
    response_model=AudiencePersonaResponse,
    summary="Run the audience persona extraction agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def audience_persona_endpoint(body: AudiencePersonaRequest) -> AudiencePersonaResponse:
    """
    Triggers the audience persona agent for an existing project.
    Reads data from strategy (company_research, visual_context_summary, brand_adjective)
    and projects (target_audience, product_details, price_and_offer, brand_guidelines),
    calls Gemini to construct an Indian audience persona, and writes the result to
    strategy.agents.audience_persona as well as logging success/failure to pipeline.agent_logs[].
    """
    logger.info("POST /audience-persona  |  project_id=%s", body.project_id)

    try:
        await run_audience_persona_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return AudiencePersonaResponse(
        message="Audience Persona agent execution finished. Check pipeline logs for detailed status."
    )


# ---------------------------------------------------------------------------
# Competitive Landscape Agent — request / response schemas
# ---------------------------------------------------------------------------

class CompetitiveLandscapeRequest(BaseModel):
    project_id: str


class CompetitiveLandscapeResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoint — POST /competitive-landscape
# ---------------------------------------------------------------------------

@app.post(
    "/competitive-landscape",
    response_model=CompetitiveLandscapeResponse,
    summary="Run the competitive landscape agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def competitive_landscape_endpoint(body: CompetitiveLandscapeRequest) -> CompetitiveLandscapeResponse:
    """
    Triggers the competitive landscape agent for an existing project.
    Reads data from strategy and project contexts, calls Gemini (with search tools)
    to perform active tracking of competitors, and writes the structured outputs to 
    strategy.agents.competitive_landscape.
    """
    logger.info("POST /competitive-landscape  |  project_id=%s", body.project_id)

    try:
        await run_competitive_landscape_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return CompetitiveLandscapeResponse(
        message="Competitive Landscape agent execution finished. Check pipeline logs for detailed status."
    )

# ---------------------------------------------------------------------------
# Central Human Truth Agent — request / response schemas
# ---------------------------------------------------------------------------

class CentralHumanTruthRequest(BaseModel):
    project_id: str


class CentralHumanTruthResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoint — POST /central-human-truth
# ---------------------------------------------------------------------------

@app.post(
    "/central-human-truth",
    response_model=CentralHumanTruthResponse,
    summary="Run the central human truth agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def central_human_truth_endpoint(body: CentralHumanTruthRequest) -> CentralHumanTruthResponse:
    """
    Triggers the central human truth agent for an existing project.
    Reads inputs: brand_adjective, audience_persona, competitive_landscape,
    product_details, company_research, and company_url.
    Calls Gemini to determine the deep emotional problem or desire
    and writes it to strategy.agents.central_human_truth.
    """
    logger.info("POST /central-human-truth  |  project_id=%s", body.project_id)

    from central_human_truth import run_central_human_truth_agent
    try:
        await run_central_human_truth_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return CentralHumanTruthResponse(
        message="Central Human Truth agent execution finished. Output is written to DB."
    )


# ---------------------------------------------------------------------------
# Value Prop & Offer Agent — request / response schemas
# ---------------------------------------------------------------------------

class ValuePropAndOfferRequest(BaseModel):
    project_id: str


class ValuePropAndOfferResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoint — POST /value-prop-and-offer
# ---------------------------------------------------------------------------

@app.post(
    "/value-prop-and-offer",
    response_model=ValuePropAndOfferResponse,
    summary="Run the value prop and offer agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def value_prop_and_offer_endpoint(body: ValuePropAndOfferRequest) -> ValuePropAndOfferResponse:
    """
    Triggers the value prop & offer agent for an existing project.
    Reads inputs: central_human_truth, audience_persona, product_details,
    price_and_offer, and company_research.
    Calls Gemini to build the rational bridge between the emotional truth and product,
    and writes it to strategy.agents.value_prop_and_offer.
    """
    logger.info("POST /value-prop-and-offer  |  project_id=%s", body.project_id)

    try:
        await run_value_prop_and_offer_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return ValuePropAndOfferResponse(
        message="Value Prop & Offer agent execution finished. Output is written to DB."
    )


# ---------------------------------------------------------------------------
# Truest Thing Agent — request / response schemas
# ---------------------------------------------------------------------------

class TruestThingRequest(BaseModel):
    project_id: str


class TruestThingResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoint — POST /truest-thing
# ---------------------------------------------------------------------------

@app.post(
    "/truest-thing",
    response_model=TruestThingResponse,
    summary="Run the truest thing agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def truest_thing_endpoint(body: TruestThingRequest) -> TruestThingResponse:
    """
    Triggers the truest thing agent for an existing project.
    Reads inputs: brand_adjective, central_human_truth, value_prop_and_offer,
    and product_details.
    Calls Gemini to define the foundational truth all messaging must adhere to,
    and writes it to strategy.agents.truest_thing.
    """
    logger.info("POST /truest-thing  |  project_id=%s", body.project_id)

    try:
        await run_truest_thing_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return TruestThingResponse(
        message="Truest Thing agent execution finished. Output is written to DB."
    )




@app.post(
    "/insight-validation",
    response_model=InsightValidationResponse,
    summary="Run the insight validation agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def insight_validation_endpoint(body: InsightValidationRequest) -> InsightValidationResponse:
    """
    Triggers the insight validation agent for an existing project.
    Reads inputs: target_audience, central_human_truth, truest_thing,
    and value_prop_and_offer.
    Calls Gemini to stress-test specificity, defensibility, and emotional resonance
    and writes the verdict to strategy.agents.insight_validation.
    """
    logger.info("POST /insight-validation  |  project_id=%s", body.project_id)

    try:
        await run_insight_validation_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return InsightValidationResponse(
        message="Insight Validation agent execution finished. Output is written to DB."
    )

# ---------------------------------------------------------------------------
# Endpoint — POST /conflict-identification
# ---------------------------------------------------------------------------

@app.post(
    "/conflict-identification",
    response_model=ConflictIdentificationResponse,
    summary="Run the conflict identification agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def conflict_identification_endpoint(body: ConflictIdentificationRequest) -> ConflictIdentificationResponse:
    """
    Triggers the conflict identification agent for an existing project.
    Reads inputs: central_human_truth, truest_thing, value_prop_and_offer,
    audience_persona, visual_context_summary, competitive_landscape, and
    product_details.
    Calls Gemini to identify the enemy and state the conflict as a dramatic
    tension, then writes it to strategy.agents.conflict_identification.
    """
    logger.info("POST /conflict-identification  |  project_id=%s", body.project_id)

    try:
        await run_conflict_identification_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return ConflictIdentificationResponse(
        message="Conflict Identification agent execution finished. Output is written to DB."
    )


# ---------------------------------------------------------------------------
# Truth + Conflict = Platform Agent
# ---------------------------------------------------------------------------

@app.post(
    "/truth-conflict-platform",
    response_model=TruthConflictPlatformResponse,
    summary="Run the truth + conflict = platform agent",
    tags=["Agents — Phase 1"],
    status_code=200,
)
async def truth_conflict_platform_endpoint(body: TruthConflictPlatformRequest) -> TruthConflictPlatformResponse:
    """
    Triggers the Truth + Conflict = Platform agent for an existing project.
    Reads data from strategy (central_human_truth, truest_thing, conflict_identification),
    calls Gemini to generate and score 3 platform candidates, and writes the selected platform to
    strategy.agents.truth_conflict_platform as well as logging reasoning to pipeline.agent_logs[].
    """
    logger.info("POST /truth-conflict-platform  |  project_id=%s", body.project_id)

    try:
        await run_truth_conflict_platform_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return TruthConflictPlatformResponse(
        message="Truth + Conflict = Platform agent execution finished. Check pipeline logs for detailed status."
    )




@app.post(
    "/strategy-models",
    response_model=StrategyModelsResponse,
    tags=["Agents — Phase 1"],
)
async def strategy_models_endpoint(body: StrategyModelsRequest):
    """
    Trigger the Strategy Models agent.
    Needs ALL previous agent outputs from the DB.
    """
    logger.info("POST /strategy-models  |  project_id=%s", body.project_id)

    try:
        await run_strategy_models_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return StrategyModelsResponse(
        message="Strategy Models agent execution finished. Check pipeline logs for detailed status."
    )

# ---------------------------------------------------------------------------
# Positioning Alignment Agent — request / response schemas
# ---------------------------------------------------------------------------
class PositioningAlignmentRequest(BaseModel):
    project_id: str

class PositioningAlignmentResponse(BaseModel):
    message: str

# ---------------------------------------------------------------------------
# Endpoint — POST /positioning-alignment
# ---------------------------------------------------------------------------
@app.post(
    "/positioning-alignment",
    response_model=PositioningAlignmentResponse,
    tags=["Agents — Phase 1"],
)
async def positioning_alignment_endpoint(body: PositioningAlignmentRequest):
    """
    Trigger the Positioning Alignment agent.
    Needs ALL previous agent outputs from the DB.
    """
    logger.info("POST /positioning-alignment  |  project_id=%s", body.project_id)

    try:
        await run_positioning_alignment_agent(
            project_id=body.project_id,
            db=_db,
        )
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return PositioningAlignmentResponse(
        message="Positioning Alignment agent execution finished. Check pipeline logs for detailed status."
    )

# ---------------------------------------------------------------------------
# Endpoint — POST /run-phase1-pipeline
# ---------------------------------------------------------------------------
@app.post(
    "/run-phase1-pipeline",
    summary="Trigger full Phase-1 LangGraph pipeline (non-blocking)",
    tags=["Pipeline"],
    status_code=202,
)
async def run_phase1_pipeline_endpoint(
    body: Phase1PipelineRequest,
    background_tasks: BackgroundTasks,
):
    """
    Kicks off the complete Phase 1 multi-agent pipeline in the background and
    returns immediately with 202. Poll GET /run-phase1-pipeline/{project_id}/status
    for live progress.
    """
    logger.info("POST /run-phase1-pipeline | project_id=%s", body.project_id)
    background_tasks.add_task(run_phase_1_pipeline, body.project_id, _db)
    return JSONResponse(
        status_code=202,
        content={"message": "Phase 1 pipeline started.", "project_id": body.project_id},
    )


# ---------------------------------------------------------------------------
# Pre-production full pipeline (Phase 1 → Phase 2 → Phase 3)
# ---------------------------------------------------------------------------
async def _run_pre_production_pipeline(project_id: str, db) -> None:
    logger.info("Pre-production pipeline started | project_id=%s", project_id)

    result1 = await run_phase_1_pipeline(project_id, db)
    if result1.get("status") != "success":
        logger.error(
            "Pre-production pipeline stopped at Phase 1 | project_id=%s | error=%s",
            project_id, result1.get("error"),
        )
        return
    logger.info("Pre-production: Phase 1 complete | project_id=%s", project_id)

    result2 = await run_phase_2_pipeline(project_id, db)
    if result2.get("status") != "completed":
        logger.error(
            "Pre-production pipeline stopped at Phase 2 | project_id=%s | error=%s",
            project_id, result2.get("error"),
        )
        return
    logger.info("Pre-production: Phase 2 complete | project_id=%s", project_id)

    result3 = await run_phase_3_pipeline(project_id, db)
    if result3.get("status") == "error":
        logger.error(
            "Pre-production pipeline failed at Phase 3 | project_id=%s", project_id
        )
        return
    logger.info("Pre-production pipeline fully complete | project_id=%s", project_id)


@app.post(
    "/run-pre-production",
    summary="Trigger full pre-production pipeline (Phase 1 → 2 → 3, non-blocking)",
    tags=["Pipeline"],
    status_code=202,
)
async def run_pre_production_endpoint(
    body: PreProductionRequest,
    background_tasks: BackgroundTasks,
):
    logger.info("POST /run-pre-production | project_id=%s", body.project_id)
    background_tasks.add_task(_run_pre_production_pipeline, body.project_id, _db)
    return JSONResponse(
        status_code=202,
        content={"message": "Pre-production pipeline started.", "project_id": body.project_id},
    )


# ---------------------------------------------------------------------------
# Ordered agent key list used by the status endpoint
# ---------------------------------------------------------------------------
_PHASE1_AGENT_ORDER = [
    "brand_adjective",
    "audience_persona",
    "competitive_landscape",
    "central_human_truth",
    "value_prop_and_offer",
    "truest_thing",
    "insight_validation",
    "conflict_identification",
    "truth_conflict_platform",
    "strategy_models",
    "positioning_alignment",
]

PIPELINE_COLLECTION: str = os.getenv("COLLECTION_PIPELINE", "pipeline")


@app.get(
    "/run-phase1-pipeline/{project_id}/status",
    summary="Poll Phase-1 pipeline progress",
    tags=["Pipeline"],
)
async def phase1_pipeline_status(project_id: str):
    """
    Returns the current progress of the Phase 1 pipeline for a given project.
    Reads pipeline.agent_logs[] from MongoDB — no Redis or Celery required.

    Response fields:
    - status: pending | running | completed | failed
    - agents_completed: ordered list of agent keys that finished successfully
    - current_agent: agent key currently executing (last one logged), or null
    - total_agents: 11
    """
    pipeline_doc = await _db[PIPELINE_COLLECTION].find_one({"project_id": project_id})
    if not pipeline_doc:
        return {
            "project_id": project_id,
            "status": "pending",
            "agents_completed": [],
            "current_agent": None,
            "total_agents": len(_PHASE1_AGENT_ORDER),
        }

    logs = pipeline_doc.get("agent_logs", [])

    completed = [
        key for key in _PHASE1_AGENT_ORDER
        if any(
            log.get("agent_key") == key and log.get("status") == "completed"
            for log in logs
        )
    ]

    failed_keys = [
        log.get("agent_key") for log in logs if log.get("status") == "failed"
    ]

    if failed_keys:
        overall_status = "failed"
    elif len(completed) == len(_PHASE1_AGENT_ORDER):
        overall_status = "completed"
    elif completed:
        overall_status = "running"
    else:
        overall_status = "pending"

    current_agent = completed[-1] if completed else None

    return {
        "project_id": project_id,
        "status": overall_status,
        "agents_completed": completed,
        "current_agent": current_agent,
        "total_agents": len(_PHASE1_AGENT_ORDER),
        "failed_agents": failed_keys,
    }

# ---------------------------------------------------------------------------
# Include Phase 2 Router (Placed at the bottom so it appears after Phase 1)
# ---------------------------------------------------------------------------
app.include_router(phase2_router, prefix="/phase-2")

# ---------------------------------------------------------------------------
# Include Phase 3 Router
# ---------------------------------------------------------------------------
app.include_router(phase3_router, prefix="/phase-3")
