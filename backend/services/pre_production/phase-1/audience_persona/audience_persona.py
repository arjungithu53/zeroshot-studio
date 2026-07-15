import os
import json
import time
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
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
logger = logging.getLogger("zeroshot.audience_persona")

router = APIRouter(prefix="/api/v1/audience-persona", tags=["Audience Persona"])

class AudiencePersonaResult(BaseModel):
    persona_name: str = Field(description="First name only, Indian name appropriate to the persona")
    age_range: str = Field(description="e.g. 24-28")
    location_context: str = Field(description="e.g. Metro Tier 1 Indian city, or Tier 2 city like Indore or Coimbatore")
    daily_life: str = Field(description="2-3 sentences describing a typical day with specific Indian texture")
    core_motivation: str = Field(description="What they are actively chasing right now in life")
    core_fear: str = Field(description="What they are quietly avoiding or anxious about")
    trigger_events: List[str] = Field(description="Specific real-life moments that make them open to buying this product")
    purchase_barriers: List[str] = Field(description="The exact reasons they would hesitate or abandon")
    media_habits: List[str] = Field(description="India-specific platforms/habits: Reels, YouTube, WhatsApp, ShareChat, etc.")
    buying_driver: str = Field(description="Single most important psychological purchase lever for this specific product")

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

async def run_audience_persona_agent(project_id: str, db):
    agent_key = "audience_persona"
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

        # Extract strategy fields
        company_research = strategy_doc.get("company_research", {})
        raw_text = company_research.get("raw_text", "")
        visual_context_summary = strategy_doc.get("visual_context_summary", "")
        
        agents_data = strategy_doc.get("agents", {})
        brand_adjective = agents_data.get("brand_adjective", "")

        # Validation
        logger.info(f"[{agent_key}] Validating required inputs.")
        if not raw_text:
            raise ValueError("raw_text is empty or missing from strategy.company_research")
        if not visual_context_summary:
            raise ValueError("visual_context_summary is empty or missing from strategy")
        if not brand_adjective:
            raise ValueError("brand_adjective is empty or missing from strategy.agents")

        # Extract project fields
        target_audience = project_doc.get("target_audience", "")
        product_details = project_doc.get("product_details", "")
        price_and_offer = project_doc.get("price_and_offer", "")
        brand_guidelines = project_doc.get("brand_guidelines", "")
        product_url = project_doc.get("product_url") or "Not provided"

        logger.info(f"[{agent_key}] Successfully loaded all required inputs.")

        # 2. Call Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")

        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"}
        )
        
        prompt = f"""You are a senior brand strategist specialising in the Indian consumer market with deep knowledge of urban and semi-urban Indian psychology, buying behaviour, and media habits.

You will be given the following inputs about a brand and its product:
1. TARGET AUDIENCE — age range, gender, provided by the brand owner
2. PRODUCT DETAILS — description of the product
3. PRICE AND OFFER — pricing and any active promotions
4. BRAND GUIDELINES — brand voice rules and restrictions (may be null)
5. COMPANY RESEARCH — raw scraped text from the brand's website
6. VISUAL CONTEXT — prose analysis of the brand's product image
7. BRAND ADJECTIVE — the single word that defines this brand's core identity

Your task is to build ONE primary audience persona. This must be a specific, believable Indian person — not a generic demographic profile.

RULES:
- Ground the persona in Indian urban or semi-urban reality
- Consider Tier 1 vs Tier 2 city context based on price_and_offer and brand cues
- The buying_driver must be a single specific psychological lever — not a generic statement like "they want value for money"
- Include daily life texture — what their morning looks like, what they scroll, what they aspire to
- core_motivation is what they are actively chasing in life right now
- core_fear is what they are quietly avoiding or anxious about
- trigger_events are specific real-life moments that make them open to buying this product
- purchase_barriers are the exact reasons they would hesitate or abandon
- media_habits must be India-specific: Reels, YouTube, WhatsApp, ShareChat, Moj, OTT platforms etc.
- If brand_guidelines are provided, ensure the persona language and tone description respects any stated brand voice rules
- Do not produce a corporate persona template. Write like you know this person.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no preamble:
{{
  "persona_name": "<first name only, Indian name appropriate to the persona>",
  "age_range": "<e.g. 24-28>",
  "location_context": "<e.g. Metro Tier 1 Indian city, or Tier 2 city like Indore or Coimbatore>",
  "daily_life": "<2-3 sentences describing a typical day with specific Indian texture>",
  "core_motivation": "<what they are actively chasing right now in life>",
  "core_fear": "<what they are quietly avoiding or anxious about>",
  "trigger_events": ["<specific moment 1>", "<specific moment 2>", "<specific moment 3>"],
  "purchase_barriers": ["<barrier 1>", "<barrier 2>", "<barrier 3>"],
  "media_habits": ["<platform/habit 1>", "<platform/habit 2>", "<platform/habit 3>"],
  "buying_driver": "<single most important psychological purchase lever for this specific product>"
}}

TARGET AUDIENCE:
{target_audience}

PRODUCT DETAILS:
{product_details}

PRODUCT WEBPAGE URL:
{product_url}
(If a product URL is provided, use it to research the specific product being advertised in depth.)

PRICE AND OFFER:
{price_and_offer}

BRAND GUIDELINES:
{brand_guidelines}

COMPANY RESEARCH:
{raw_text}

VISUAL CONTEXT:
{visual_context_summary}

BRAND ADJECTIVE:
{brand_adjective}
"""
        logger.info(f"[{agent_key}] Calling Gemini model: {GEMINI_MODEL}")
        start_time = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "tools": [{"google_search": {}}],
                "response_mime_type": "application/json",
                "response_json_schema": AudiencePersonaResult.model_json_schema(),
            }
        )
        api_duration = time.time() - start_time

        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")
        
        # 3. Parse output
        logger.info(f"[{agent_key}] Parsing and cleaning JSON response.")
        cleaned_json_str = clean_json_string(response.text)
        try:
            persona_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as je:
            raise ValueError(f"Failed to parse JSON from Gemini response: {je}. Raw response: {response.text}")

        persona_name = persona_data.get("persona_name", "Unknown")

        # 4. Save to db (Strategy Collection)
        logger.info(f"[{agent_key}] Writing generated persona to strategy.agents.audience_persona.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": {"agents.audience_persona": persona_data}}
        )

        # 5. Log success to pipeline
        logger.info(f"[{agent_key}] Writing success log to pipeline document.")
        success_log = {
            "agent_key": agent_key,
            "status": "completed",
            "persona_name": persona_name,
            "timestamp": timestamp
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": success_log}},
            upsert=True
        )

        logger.info(f"[{agent_key}] Execution completed successfully.")

    except Exception as e:
        logger.error(f"[{agent_key}] Gracefully failing agent due to error: {str(e)}", exc_info=True)
        # Log failure to pipeline collection and do not raise
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

