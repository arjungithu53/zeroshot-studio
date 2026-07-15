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
from typing import List

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
PIPELINE_COLLECTION  = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION  = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION  = os.getenv("COLLECTION_PROJECTS", "projects")

# Setup logger
logger = logging.getLogger("zeroshot.value_prop_and_offer")

class ValuePropAndOfferResult(BaseModel):
    primary_benefit: str = Field(description="Concrete rational benefit that directly solves the emotional problem")
    secondary_benefits: List[str] = Field(description="Max 3 secondary rational benefits", max_length=3)
    offer_hook: str = Field(description="Lowest-friction, highest-perceived-value entry point, e.g. Risk-free 30-day trial or Rs.199 starter kit")
    barrier_addressed: str = Field(description="Which persona purchase barrier this offer resolves")
    rational_bridge: str = Field(description="1 sentence connecting emotional truth to rational offer")
    reasoning: str = Field(description="Explanation of why this value prop and offer is irresistible to the persona")

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

async def run_value_prop_and_offer_agent(project_id: str, db):
    agent_key = "value_prop_and_offer"
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
        company_research = strategy_doc.get("company_research", {})
        raw_text = company_research.get("raw_text", "")
        
        agents_data = strategy_doc.get("agents", {})
        central_human_truth = agents_data.get("central_human_truth", {})
        audience_persona = agents_data.get("audience_persona", {})

        product_details = project_doc.get("product_details", "")
        price_and_offer = project_doc.get("price_and_offer", "")
        product_url = project_doc.get("product_url") or "Not provided"

        # Validation
        logger.info(f"[{agent_key}] Validating required inputs.")
        missing_inputs = []
        if not central_human_truth: missing_inputs.append("central_human_truth")
        if not audience_persona: missing_inputs.append("audience_persona")
        if not product_details: missing_inputs.append("product_details")
        if not raw_text: missing_inputs.append("company_research.raw_text")
        
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

        prompt = f"""You are a master Conversion Copywriter and Performance Marketer identifying the core "Value Prop & Offer".
Your goal is to build the rational bridge between the deep emotional truth (Agent 4) and the product's actual features and pricing.

RULES:
1. Translate the product features and pricing/offer into concrete rational benefits that directly solve the emotional problem identified.
2. The emotional hook gets them to stop scrolling. The value prop gets them to click and buy.
3. Identify the single most irresistible offer hook: the lowest-friction, highest-perceived-value entry point for this persona.
4. Consider purchase_barriers from the Audience Persona — the offer framing should directly address the most critical barrier.
5. If price_and_offer is null or weak, derive an implied value or starter offer from product_details and company website text.

INPUTS:

Central Human Truth:
{json.dumps(central_human_truth, indent=2)}

Audience Persona:
{json.dumps(audience_persona, indent=2)}

Product Details:
{product_details}

Product Webpage URL:
{product_url}
(If a product URL is provided, use it to research the specific product's features and benefits in depth.)

Pricing and Offer Details:
{price_and_offer}

Company Research (Raw Text):
{raw_text}

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no preamble:"""

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
                "response_json_schema": ValuePropAndOfferResult.model_json_schema(),
            }
        )
        api_duration = time.time() - start_time
        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")

        # 3. Parse output
        logger.info(f"[{agent_key}] Parsing and cleaning JSON response.")
        cleaned_json_str = clean_json_string(response.text)
        try:
            truth_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as je:
            raise ValueError(f"Failed to parse JSON from Gemini response: {je}. Raw response: {response.text}")

        reasoning = truth_data.pop("reasoning", "")

        # 4. Save to db
        logger.info(f"[{agent_key}] Writing generated value prop to strategy.agents.value_prop_and_offer.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": {f"agents.{agent_key}": truth_data}}
        )

        # 5. Log success to pipeline
        logger.info(f"[{agent_key}] Writing success log to pipeline document.")
        success_log = {
            "agent_key": agent_key,
            "status": "completed",
            "reasoning": reasoning,
            "timestamp": timestamp
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": success_log}},
            upsert=True
        )

        logger.info(f"[{agent_key}] Execution completed successfully.")
        return truth_data

    except Exception as e:
        logger.error(f"[{agent_key}] Gracefully failing agent due to error: {str(e)}", exc_info=True)
        error_log = {
            "agent_key": agent_key,
            "status": "failed",
            "error_message": str(e),
            "timestamp": datetime.now(timezone.utc)
        }
        try:
            await db[PIPELINE_COLLECTION].update_one(
                {"project_id": str(project_id)},
                {"$push": {"agent_logs": error_log}},
                upsert=True
            )
            logger.info(f"[{agent_key}] Error successfully logged to pipeline document.")
        except Exception as db_e:
            logger.error(f"[{agent_key}] Critical failure: Could not log error to pipeline: {str(db_e)}")
