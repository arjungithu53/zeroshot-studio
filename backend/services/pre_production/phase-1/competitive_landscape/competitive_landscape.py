import os
import json
import time
import logging
from datetime import datetime, timezone
from fastapi import APIRouter
from bson import ObjectId
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
PIPELINE_COLLECTION  = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION  = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION  = os.getenv("COLLECTION_PROJECTS", "projects")

# Setup logger
logger = logging.getLogger("zeroshot.competitive_landscape")

router = APIRouter(prefix="/api/v1/competitive-landscape", tags=["Competitive Landscape"])

class Competitor(BaseModel):
    name: str = Field(description="Name of the competitor")
    dominant_message: str = Field(description="The main message they communicate")
    tone: str = Field(description="The tone of their messaging")

class CompetitiveLandscapeResult(BaseModel):
    product_category: str = Field(description="The category of the product")
    competitors: List[Competitor] = Field(description="List of 3 to 5 real competitors")
    category_dominant_tone: str = Field(description="The dominant tone used across the category")
    repeated_messages: List[str] = Field(description="What the persona is exhausted by hearing repeatedly")
    whitespace_opportunity: str = Field(description="What the category is NOT saying that the persona wants to hear")
    conventions_to_break: List[str] = Field(description="Category conventions this brand should break")
    conventions_to_follow: List[str] = Field(description="Category conventions this brand should follow")
    fallback: bool = Field(description="True if Search grounding was unavailable, False otherwise")

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

async def run_competitive_landscape_agent(project_id: str, db):
    agent_key = "competitive_landscape"
    
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
        
        agents_data = strategy_doc.get("agents", {})
        brand_adjective = agents_data.get("brand_adjective", "")
        audience_persona = agents_data.get("audience_persona", {})

        # Validation
        logger.info(f"[{agent_key}] Validating required inputs.")
        if not raw_text:
            raise ValueError("raw_text is empty or missing from strategy.company_research")
        if not brand_adjective:
            raise ValueError("brand_adjective is empty or missing from strategy.agents")
        if not audience_persona:
            raise ValueError("audience_persona is empty or missing from strategy.agents")

        # Extract project fields
        product_details = project_doc.get("product_details", "")
        if not product_details:
            raise ValueError("product_details is empty or missing from project document")
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
        
        prompt = f"""You are a senior brand strategist specializing in analyzing competitive landscapes in the Indian market.

You are equipped with Google Search capabilities to actively research the product category and specific competitors.

Inputs available:
1. PRODUCT DETAILS:
{product_details}
2. PRODUCT WEBPAGE URL:
{product_url}
(If a product URL is provided, use it to research the specific product in depth before analyzing the competitive landscape.)
3. COMPANY RESEARCH:
{raw_text}
4. BRAND ADJECTIVE:
{brand_adjective}
5. AUDIENCE PERSONA:
{json.dumps(audience_persona)}

Your task:
- Use Google Search to actively research: Top brands in this product category in India, competitors for this specific client brand in India, and category-specific ad messaging patterns on Meta and Instagram.
- Identify 3 to 5 real competitors and summarize their dominant messaging, tone, and visual codes.
- Analyze what the target persona (from the audience persona input) is being told repeatedly by the category — what are they exhausted by hearing?
- Identify the whitespace: what is the category NOT saying that this persona deeply wants to hear?
- Flag which category conventions should be broken vs. followed for this specific brand.
- If your Google Search attempts fail or you cannot get reliable online data, fallback to your own knowledge and set fallback = true in the output. Otherwise, set fallback = false.

OUTPUT FORMAT — respond ONLY with valid JSON matching the exact schema requested, no markdown, no preamble:
"""

        logger.info(f"[{agent_key}] Calling Gemini model: {GEMINI_MODEL}")
        start_time = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "tools": [{"google_search": {}}],
                "response_mime_type": "application/json",
                "response_json_schema": CompetitiveLandscapeResult.model_json_schema(),
            }
        )
        api_duration = time.time() - start_time

        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")
        
        # 3. Parse output
        logger.info(f"[{agent_key}] Parsing and cleaning JSON response.")
        cleaned_json_str = clean_json_string(response.text)
        try:
            landscape_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as je:
            raise ValueError(f"Failed to parse JSON from Gemini response: {je}. Raw response: {response.text}")

        # 4. Save to db (Strategy Collection)
        logger.info(f"[{agent_key}] Writing generated landscape to strategy.agents.competitive_landscape.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": {"agents.competitive_landscape": landscape_data}}
        )

        return landscape_data

    except Exception as e:
        logger.error(f"[{agent_key}] Error: {str(e)}")
        raise
