import os
import json
import time
import logging
from datetime import datetime, timezone
from bson import ObjectId
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
PIPELINE_COLLECTION  = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION  = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION  = os.getenv("COLLECTION_PROJECTS", "projects")

# Setup logger
logger = logging.getLogger("zeroshot.insight_validation")

# --- Pydantic Data Models (Outputs) ---

class CheckNote(BaseModel):
    pass_status: bool = Field(alias="pass", description="Did this check pass?")
    note: str = Field(description="Explanation of why it passed or failed")

class ValidationChecks(BaseModel):
    specificity: CheckNote = Field(description="Could this truth apply to a competitor brand?")
    defensibility: CheckNote = Field(description="Can the product actually back this claim?")
    resonance: CheckNote = Field(description="Does it trigger a felt response, not just intellectual agreement?")

class InsightValidationResult(BaseModel):
    validation_status: str = Field(description="Must be exactly 'validated' or 'flagged'")
    checks: ValidationChecks = Field(description="Validation checks for specificity, defensibility, and resonance")
    suggested_correction: Optional[str] = Field(description="If flagged, a specific suggested correction. Null if validated.")


def clean_json_string(raw_str: str) -> str:
    """Helper to strip markdown fences from Gemini JSON response."""
    cleaned = raw_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

async def run_insight_validation_agent(project_id: str, db):
    agent_key = "insight_validation"
    timestamp = datetime.now(timezone.utc)
    
    logger.info(f"[{agent_key}] Starting agent. project_id={project_id}")
    
    try:
        # 1. Fetch data from DB
        logger.info(f"[{agent_key}] Fetching strategy document.")
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": project_id})
        if not strategy_doc:
            raise ValueError(f"Strategy document for project {project_id} not found.")
            
        strategy_id = strategy_doc["_id"]

        logger.info(f"[{agent_key}] Fetching project document.")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project document {project_id} not found.")

        # Extract fields
        agents_data = strategy_doc.get("agents", {})
        central_human_truth = agents_data.get("central_human_truth", {})
        truest_thing = agents_data.get("truest_thing", {})
        value_prop_and_offer = agents_data.get("value_prop_and_offer", {})

        target_audience = project_doc.get("target_audience", "")
        if not target_audience and "audience_persona" in agents_data:
            target_audience = agents_data["audience_persona"]

        # Validation
        logger.info(f"[{agent_key}] Validating required inputs.")
        missing_inputs = []
        if not target_audience: missing_inputs.append("target_audience")
        if not central_human_truth: missing_inputs.append("central_human_truth")
        if not truest_thing: missing_inputs.append("truest_thing")
        if not value_prop_and_offer: missing_inputs.append("value_prop_and_offer")
        
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {', '.join(missing_inputs)}")

        logger.info(f"[{agent_key}] Successfully loaded all required inputs.")

        # 2. Call Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")

        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"}
        )

        prompt = f"""You are a rigorous Brand Strategy Stress-Tester.
Your goal is to validate the core insights discovered in earlier phases (Central Human Truth, Truest Thing, Value Prop & Offer) against the Target Audience.

RULES:
1. Run three validation checks on these insights:
   (1) Specificity — Could this truth apply to a competitor brand? If it is generic, it fails.
   (2) Defensibility — Can the product's value prop and offer actually back this claim? If the product cannot deliver, it fails.
   (3) Emotional resonance — Does the truth trigger a felt response, not just intellectual agreement, from the target audience? If boring or purely logical, it fails.
2. If all three pass: validation_status = 'validated', suggested_correction = null.
3. If ANY fail: validation_status = 'flagged', explain why in the corresponding note, and provide a suggested_correction string.

INPUTS:

Target Audience:
{json.dumps(target_audience, indent=2) if isinstance(target_audience, dict) else target_audience}

Central Human Truth:
{json.dumps(central_human_truth, indent=2)}

Truest Thing:
{json.dumps(truest_thing, indent=2)}

Value Prop & Offer:
{json.dumps(value_prop_and_offer, indent=2)}

OUTPUT FORMAT — respond ONLY with valid JSON matching the schema, no markdown, no preamble:"""

        logger.info(f"[{agent_key}] Calling Gemini model: {GEMINI_MODEL}")
        start_time = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "tools": [
                    {"google_search": {}},
                    {"url_context": {}},
                ],
                "response_mime_type": "application/json",
                "response_json_schema": InsightValidationResult.model_json_schema(),
            }
        )
        api_duration = time.time() - start_time
        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")

        # 3. Parse output
        logger.info(f"[{agent_key}] Parsing and cleaning JSON response.")
        cleaned_json_str = clean_json_string(response.text)
        try:
            result_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as je:
            raise ValueError(f"Failed to parse JSON from Gemini response: {je}. Raw response: {response.text}")

        # 4. Save to DB
        logger.info(f"[{agent_key}] Writing output to Strategy collection.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": strategy_id},
            {"$set": {f"agents.{agent_key}": result_data}}
        )

        # 5. Pipeline Logging
        logger.info(f"[{agent_key}] Logging success to Pipeline collection.")
        success_log = {
            "agent_key": agent_key,
            "status": "completed",
            "validation_status": result_data.get("validation_status"),
            "timestamp": timestamp,
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": success_log}},
            upsert=True
        )

        return result_data

    except Exception as exc:
        logger.error(f"[{agent_key}] Gracefully failing agent due to error: {exc}", exc_info=True)
        error_log = {
            "agent_key": agent_key,
            "status": "failed",
            "error_message": str(exc),
            "timestamp": datetime.now(timezone.utc)
        }
        try:
            await db[PIPELINE_COLLECTION].update_one(
                {"project_id": str(project_id)},
                {"$push": {"agent_logs": error_log}},
                upsert=True
            )
        except Exception as logger_exc:
            logger.error(f"Failed to write error log to pipeline: {logger_exc}")
        
        raise ValueError(str(exc))
