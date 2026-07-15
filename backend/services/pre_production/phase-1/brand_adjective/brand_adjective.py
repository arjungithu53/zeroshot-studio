"""
Phase 1 Agent — Brand Adjective Extraction

Reads crawl output and visual context from MongoDB, calls Gemini with
google_search + url_context tools to identify the single most dominant
brand adjective, then writes the result back to MongoDB.

Reads from MongoDB:
  strategy  — company_research.raw_text, visual_context_summary
  projects  — product_details, company_url

Writes to MongoDB:
  strategy  — agents.brand_adjective  (single word string)
  pipeline  — agent_logs[]            (full log: reasoning, status, timestamp)

Raises on all errors — the caller (FastAPI endpoint) is responsible for
handling exceptions and returning appropriate HTTP responses.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger("zeroshot.brand_adjective")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
PIPELINE_COLLECTION  = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION  = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION  = os.getenv("COLLECTION_PROJECTS", "projects")

PROMPT_TEMPLATE = """\
Act as an elite brand strategist operating under the mental model of "draconian simplicity." In the mind of the consumer, a brand equals a single adjective. Your objective is to strip a brand's value proposition down to the marrow, carving away the clutter to claim one definitive square foot of mental real estate.

Execute the following brand analysis workflow:

SYNTHESIS: Analyze the provided text sources and utilize the live website to map the brand's complete messaging ecosystem.

REDUCTION: Identify and discard secondary benefits (e.g., "lasts longer", "less expensive"). These are clutter that belong in brochures, not the core identity.

ISOLATION: Define the core emotional or functional anchor. Determine the single word a casual customer would use to describe the brand in conversation (e.g., Jeep = tough, Volvo = safe).

FLANKING: If the obvious adjective is a generic category expectation, execute a flanking move. Select a striking, polar-opposite, or non-traditional descriptor (e.g., Heinz claiming "slowest" instead of "tomato-iest").

CONSTRAINTS & RULES:

Ground all deductions strictly in the provided data sources and live URL context.

Target a sharp, emotionally resonant register (e.g., bold, nurturing, irreverent, grounded, clinical, playful).

Exclude the brand name from consideration.

Exclude generic category descriptors (e.g., premium, quality, innovative, trusted, refreshing).

Limit the brand adjective to exactly one word. Omit hyphens and compound adjectives entirely.

Verify your chosen adjective using specific evidence from at least two different data sources.

OUTPUT FORMAT:
Respond ONLY with valid JSON. Do not include markdown formatting or preamble. Use the exact schema below:
{{
"brand_adjective": "<single word>",
"reasoning": "<2-3 sentences citing specific evidence from crawl output, visual context, or live website>"
}}

DATA SOURCES:

LIVE WEBSITE URL:
{company_url}

PRODUCT WEBPAGE URL:
{product_url}
(If a product URL is provided, use it to understand the specific product being advertised in depth.)

CRAWL OUTPUT:
{raw_text}

VISUAL CONTEXT:
{visual_context_summary}

PRODUCT DETAILS:
{product_details}
"""


# ---------------------------------------------------------------------------
# Pydantic schema for structured Gemini output
# ---------------------------------------------------------------------------

class BrandAdjectiveResult(BaseModel):
    brand_adjective: str = Field(description="The single most dominant brand adjective.")
    reasoning: str = Field(
        description=(
            "5-6 sentences citing specific evidence from crawl output, "
            "visual context, or live website."
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` wrappers if Gemini adds them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def _push_failure_log(
    db: AsyncIOMotorDatabase,
    project_id: str,
    error: str,
) -> None:
    """Best-effort pipeline failure log — called before re-raising."""
    try:
        log_entry = {
            "agent_key":  "brand_adjective",
            "status":     "failed",
            "error":      error,
            "timestamp":  datetime.now(timezone.utc),
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": project_id},
            {"$push": {"agent_logs": log_entry}},
        )
        logger.info(
            "[BRAND_ADJECTIVE] Failure log written  |  project_id=%s  error=%s",
            project_id, error,
        )
    except Exception as log_exc:
        logger.warning(
            "[BRAND_ADJECTIVE] Could not write failure log  |  project_id=%s  log_error=%s",
            project_id, log_exc,
        )


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

async def run_brand_adjective_agent(
    project_id: str,
    db: AsyncIOMotorDatabase,
) -> BrandAdjectiveResult:
    """
    Run the brand adjective extraction agent.

    Reads from MongoDB, calls Gemini with google_search + url_context tools,
    and writes the result back to MongoDB.

    Raises:
        ValueError   — missing / empty required input fields
        RuntimeError — Gemini API failures, empty/unparseable responses,
                       or MongoDB write errors
    """
    logger.info("[BRAND_ADJECTIVE] Starting  |  project_id=%s", project_id)

    # ── 1. Read strategy document (looked up by project_id) ────────────────
    strategy_doc = await db[STRATEGY_COLLECTION].find_one(
        {"project_id": project_id}
    )
    if not strategy_doc:
        error = f"Strategy document not found  |  project_id={project_id}"
        await _push_failure_log(db, project_id, error)
        raise ValueError(error)

    raw_text       = (strategy_doc.get("company_research") or {}).get("raw_text") or ""
    visual_context = strategy_doc.get("visual_context_summary") or ""

    logger.info(
        "[BRAND_ADJECTIVE] Strategy read  |  project_id=%s  raw_text_len=%d  visual_context_len=%d",
        project_id, len(raw_text), len(visual_context),
    )

    # ── 2. Validate required inputs ────────────────────────────────────────
    if not raw_text.strip():
        error = "company_research.raw_text is empty — run company research agent first"
        await _push_failure_log(db, project_id, error)
        raise ValueError(error)

    if not visual_context.strip():
        error = "visual_context_summary is empty — run visual context agent first"
        await _push_failure_log(db, project_id, error)
        raise ValueError(error)

    # ── 3. Read project document ───────────────────────────────────────────
    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
    if not project_doc:
        error = f"Project document not found  |  project_id={project_id}"
        await _push_failure_log(db, project_id, error)
        raise ValueError(error)

    product_details = project_doc.get("product_details") or ""
    company_url     = project_doc.get("company_url") or ""
    product_url     = project_doc.get("product_url") or "Not provided"

    logger.info(
        "[BRAND_ADJECTIVE] Project read  |  project_id=%s  company_url=%s",
        project_id, company_url,
    )

    # ── 4. Build prompt ────────────────────────────────────────────────────
    prompt = PROMPT_TEMPLATE.format(
        company_url=company_url,
        product_url=product_url,
        raw_text=raw_text,
        visual_context_summary=visual_context,
        product_details=product_details,
    )

    # ── 5. Call Gemini ─────────────────────────────────────────────────────
    logger.info(
        "[BRAND_ADJECTIVE] Gemini call starting  |  model=%s  project_id=%s",
        GEMINI_MODEL, project_id,
    )

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha"},
        )
        
        start_time = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "tools": [
                    {"google_search": {}},
                    {"url_context": {}},
                ],
                "response_mime_type":    "application/json",
                "response_json_schema":  BrandAdjectiveResult.model_json_schema(),
            },
        )
        api_duration = time.time() - start_time
        logger.info("[BRAND_ADJECTIVE] Gemini API call took %.2f seconds", api_duration)
        
    except Exception as exc:
        error = f"Gemini API call failed: {exc}"
        await _push_failure_log(db, project_id, error)
        raise RuntimeError(error) from exc

    # ── 6. Extract and parse response ─────────────────────────────────────
    response_text = (response.text or "").strip()
    if not response_text:
        error = "Gemini returned an empty response"
        await _push_failure_log(db, project_id, error)
        raise RuntimeError(error)

    logger.info(
        "[BRAND_ADJECTIVE] Gemini response received  |  project_id=%s  response_len=%d",
        project_id, len(response_text),
    )

    cleaned = _strip_markdown_fences(response_text)

    try:
        result = BrandAdjectiveResult.model_validate_json(cleaned)
    except Exception as exc:
        error = f"Failed to parse Gemini JSON response: {exc}  |  raw={cleaned[:200]}"
        await _push_failure_log(db, project_id, error)
        raise RuntimeError(error) from exc

    logger.info(
        "[BRAND_ADJECTIVE] Parsed result  |  project_id=%s  brand_adjective=%s",
        project_id, result.brand_adjective,
    )

    # ── 7. Write to strategy ───────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    try:
        await db[STRATEGY_COLLECTION].update_one(
            {"project_id": project_id},
            {"$set": {"agents.brand_adjective": result.brand_adjective, "updated_at": now}},
        )
        logger.info(
            "[BRAND_ADJECTIVE] Strategy updated  |  project_id=%s  brand_adjective=%s",
            project_id, result.brand_adjective,
        )
    except Exception as exc:
        error = f"MongoDB strategy write failed: {exc}"
        await _push_failure_log(db, project_id, error)
        raise RuntimeError(error) from exc

    # ── 8. Write success log to pipeline ──────────────────────────────────
    try:
        log_entry = {
            "agent_key":       "brand_adjective",
            "brand_adjective": result.brand_adjective,
            "reasoning":       result.reasoning,
            "status":          "completed",
            "timestamp":       now,
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": project_id},
            {"$push": {"agent_logs": log_entry}},
        )
        logger.info(
            "[BRAND_ADJECTIVE] Pipeline log written  |  project_id=%s  status=completed",
            project_id,
        )
    except Exception as exc:
        # Pipeline log failure is non-fatal — result is already persisted to strategy
        logger.warning(
            "[BRAND_ADJECTIVE] Pipeline log write failed (non-fatal)  |  project_id=%s  error=%s",
            project_id, exc,
        )

    logger.info("[BRAND_ADJECTIVE] Done  |  project_id=%s", project_id)
    return result
