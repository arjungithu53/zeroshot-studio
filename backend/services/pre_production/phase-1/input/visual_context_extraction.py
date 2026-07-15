
"""
Pre-Processing Step 2 — Visual Context Extraction

Downloads the product image from S3, analyses it with Gemini 3.1 Pro (vision),
validates the response, and stores the result in MongoDB.

Writes to MongoDB:
  pipeline   — pushes to pre_processing[]: step, analised_at, error, total_words
  strategy   — sets visual_context_summary (string or null)

Never raises — all errors are caught, logged, and stored.
"""

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from botocore.exceptions import BotoCoreError, ClientError
from bson import ObjectId
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorDatabase

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("pipeline.visual_context")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "zeroshot-v1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
MIN_WORD_COUNT = 150

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

REFUSAL_PATTERNS = ["I cannot", "I'm unable", "I don't see"]

ANALYSIS_PROMPT = (
    "Analyse this product image across four dimensions: "
    "(1) Color palette and visual mood — describe dominant colors specifically, "
    "overall mood, warm or cool tone, emotions evoked. "
    "(2) Packaging style and material cues — type of packaging, apparent materials, "
    "minimal or ornate, label design communication. "
    "(3) Brand tone implied by visual design alone — what type of brand this appears "
    "to be, implied target customer, where it would be sold, any cultural or "
    "geographic visual coding. "
    "(4) Emotional promise communicated visually before reading any text — what the "
    "product promises, aspirational identity implied, whether it creates desire, "
    "trust, curiosity or reassurance, and what a standalone ad with no copy would "
    "communicate. Write a single dense prose paragraph of 150 to any number of max "
    "words. Do not use bullet points, headers or numbered lists. Do not begin with "
    "'This image shows' or 'The product is'. Begin directly with the most striking "
    "visual observation. Be specific and concrete. Use language an experienced "
    "creative director would use in a brief. Write in present tense."
)

# ---------------------------------------------------------------------------
# Extension → MIME type mapping (fallback when magic bytes are ambiguous)
# ---------------------------------------------------------------------------
_EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_mime_type(data: bytes, filename: str) -> str | None:
    """
    Detect image MIME type from magic bytes first, then fall back to
    the file extension. Returns None if the format is unsupported.
    """
    # JPEG: starts with FF D8 FF
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    # PNG: 8-byte signature
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # WebP: RIFF....WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # GIF: GIF87a or GIF89a
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"

    # Fallback to extension
    ext = os.path.splitext(filename or "")[-1].lower()
    mime = _EXT_TO_MIME.get(ext)
    if mime and mime in SUPPORTED_MIME_TYPES:
        return mime

    return None


def _validate_response(text: str) -> tuple[bool, str]:
    """
    Validate the Gemini vision response.
    Returns (passed, reason) — reason is empty string on success.
    """
    # Empty / whitespace
    if not text or not text.strip():
        return False, "Response is empty or whitespace only"

    # Word count
    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        return False, f"Word count {word_count} is below minimum {MIN_WORD_COUNT}"

    # Refusal patterns (case-insensitive)
    text_lower = text.lower()
    for pattern in REFUSAL_PATTERNS:
        if pattern.lower() in text_lower:
            return False, f"Response contains refusal pattern: '{pattern}'"

    return True, ""


async def _record_failure(
    db: AsyncIOMotorDatabase,
    project_id: str,
    strategy_id: str | None,
    error_msg: str,
) -> None:
    """
    DRY helper — on any failure, update both strategy and pipeline
    collections. Never raises; swallows its own DB errors.
    """
    now = datetime.now(timezone.utc)

    # Update strategy: set visual_context_summary to null
    if strategy_id:
        try:
            await db[STRATEGY_COLLECTION].update_one(
                {"_id": ObjectId(strategy_id)},
                {"$set": {
                    "visual_context_summary": None,
                    "updated_at": now,
                }},
            )
        except Exception as exc:
            logger.warning(
                "[PRE-PROCESSING 2] Failed to update strategy on error  |  "
                "strategy_id=%s  error=%s", strategy_id, exc,
            )

    # Push failure entry to pipeline.pre_processing[]
    try:
        log_entry = {
            "step": "Product_analysis",
            "analised_at": now,
            "error": error_msg,
            "total_words": 0,
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": project_id},
            {"$push": {"pre_processing": log_entry}},
        )
    except Exception as exc:
        logger.warning(
            "[PRE-PROCESSING 2] Failed to write pipeline log on error  |  "
            "project_id=%s  error=%s", project_id, exc,
        )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def run_visual_context_extraction(
    project_id: str,
    strategy_id: str,
    db: AsyncIOMotorDatabase,
    s3_client,
) -> str | None:
    """
    Download the product image from S3, analyse it with Gemini vision,
    validate the output, and store the result in MongoDB.

    Returns the visual_context_summary string on success, or None on failure.
    Never raises — the pipeline must continue.
    """
    logger.info(
        "[PRE-PROCESSING 2] Starting  |  project_id=%s  strategy_id=%s",
        project_id, strategy_id,
    )

    try:
        # ── 1. Fetch project document from MongoDB ─────────────────────────
        try:
            project = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        except Exception as exc:
            logger.warning(
                "[PRE-PROCESSING 2] MongoDB read failed  |  project_id=%s  error=%s",
                project_id, exc,
            )
            await _record_failure(db, project_id, strategy_id, f"MongoDB read error: {exc}")
            return None

        if not project:
            logger.warning(
                "[PRE-PROCESSING 2] Project not found  |  project_id=%s", project_id,
            )
            await _record_failure(db, project_id, strategy_id, "Project document not found")
            return None

        # ── 2. Extract S3 key and filename ─────────────────────────────────
        product_image = project.get("product_image")
        if not product_image or not product_image.get("s3_key"):
            logger.warning(
                "[PRE-PROCESSING 2] product_image field missing or null  |  "
                "project_id=%s", project_id,
            )
            await _record_failure(
                db, project_id, strategy_id,
                "product_image field missing or null in project document",
            )
            return None

        s3_key = product_image["s3_key"]
        original_filename = product_image.get("original_filename", "")

        # ── 3. Download image bytes from S3 via boto3 get_object ───────────
        try:
            s3_response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            image_bytes = s3_response["Body"].read()
            size_kb = len(image_bytes) / 1024
            logger.info(
                "[PRE-PROCESSING 2] S3 download OK  |  key=%s  size=%.1f KB",
                s3_key, size_kb,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.warning(
                "[PRE-PROCESSING 2] S3 download failed  |  key=%s  error=%s",
                s3_key, exc,
            )
            await _record_failure(db, project_id, strategy_id, f"S3 download error: {exc}")
            return None

        # ── 4. Detect image MIME type ──────────────────────────────────────
        mime_type = _detect_mime_type(image_bytes, original_filename)
        if not mime_type or mime_type not in SUPPORTED_MIME_TYPES:
            logger.warning(
                "[PRE-PROCESSING 2] Unsupported image format  |  "
                "filename=%s  detected_mime=%s", original_filename, mime_type,
            )
            await _record_failure(
                db, project_id, strategy_id,
                f"Unsupported image format: {mime_type or 'unknown'}",
            )
            return None

        # ── 5. Call Gemini vision API ──────────────────────────────────────
        logger.info("[PRE-PROCESSING 2] Gemini API call starting  |  model=%s", GEMINI_MODEL)
        try:
            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY", ""),
                http_options={"api_version": "v1alpha"},
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=ANALYSIS_PROMPT),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type,
                                    data=image_bytes,
                                ),
                                media_resolution={"level": "media_resolution_high"},
                            ),
                        ]
                    )
                ],
            )
        except Exception as exc:
            logger.warning(
                "[PRE-PROCESSING 2] Gemini API call failed  |  project_id=%s  error=%s",
                project_id, exc,
            )
            await _record_failure(db, project_id, strategy_id, f"Gemini API error: {exc}")
            return None

        # ── 6. Extract text response ───────────────────────────────────────
        response_text = (response.text or "").strip() if response else ""

        if not response_text:
            logger.warning(
                "[PRE-PROCESSING 2] Gemini returned empty response  |  project_id=%s",
                project_id,
            )
            await _record_failure(db, project_id, strategy_id, "Gemini returned empty response")
            return None

        word_count = len(response_text.split())
        logger.info(
            "[PRE-PROCESSING 2] Gemini response received  |  word_count=%d", word_count,
        )

        # ── 7. Validate the response ──────────────────────────────────────
        passed, reason = _validate_response(response_text)
        if not passed:
            logger.warning(
                "[PRE-PROCESSING 2] Validation failed  |  project_id=%s  reason=%s",
                project_id, reason,
            )
            await _record_failure(db, project_id, strategy_id, f"Validation failed: {reason}")
            return None

        logger.info("[PRE-PROCESSING 2] Validation passed  |  project_id=%s", project_id)

        # ── 8. Update strategy document in MongoDB ─────────────────────────
        now = datetime.now(timezone.utc)
        try:
            await db[STRATEGY_COLLECTION].update_one(
                {"_id": ObjectId(strategy_id)},
                {"$set": {
                    "visual_context_summary": response_text,
                    "updated_at": now,
                }},
            )
            logger.info(
                "[PRE-PROCESSING 2] Strategy updated  |  strategy_id=%s", strategy_id,
            )
        except Exception as exc:
            logger.warning(
                "[PRE-PROCESSING 2] MongoDB strategy update failed  |  "
                "strategy_id=%s  error=%s", strategy_id, exc,
            )
            await _record_failure(db, project_id, strategy_id, f"MongoDB update error: {exc}")
            return None

        # ── 9. Push success entry to pipeline.pre_processing[] ─────────────
        try:
            log_entry = {
                "step": "Product_analysis",
                "analised_at": now,
                "error": None,
                "total_words": word_count,
            }
            await db[PIPELINE_COLLECTION].update_one(
                {"project_id": project_id},
                {"$push": {"pre_processing": log_entry}},
            )
            logger.info(
                "[PRE-PROCESSING 2] Pipeline log written  |  project_id=%s  words=%d",
                project_id, word_count,
            )
        except Exception as exc:
            # Pipeline logging failure is non-fatal — the analysis succeeded
            logger.warning(
                "[PRE-PROCESSING 2] Pipeline log write failed  |  "
                "project_id=%s  error=%s", project_id, exc,
            )

        # ── 10. Done ──────────────────────────────────────────────────────
        logger.info(
            "[PRE-PROCESSING 2] Complete  |  project_id=%s  result=success", project_id,
        )
        return response_text

    except Exception as exc:
        # Catch-all — any unexpected exception
        logger.warning(
            "[PRE-PROCESSING 2] Unexpected error  |  project_id=%s  error=%s",
            project_id, exc,
        )
        await _record_failure(db, project_id, strategy_id, f"Unexpected error: {exc}")
        return None
