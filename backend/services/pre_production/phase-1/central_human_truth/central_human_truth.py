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

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
PIPELINE_COLLECTION  = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION  = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION  = os.getenv("COLLECTION_PROJECTS", "projects")

# Setup logger
logger = logging.getLogger("zeroshot.central_human_truth")

class CentralHumanTruthResult(BaseModel):
    human_truth: str = Field(description="1-2 sentence statement expressing the deep emotional problem or desire")
    audience_lens: str = Field(description="Specific sub-segment this truth applies to")
    reasoning: str = Field(description="Explanation of why this truth is powerful and untouched by competitors")

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

async def run_central_human_truth_agent(project_id: str, db):
    agent_key = "central_human_truth"
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
        brand_adjective = agents_data.get("brand_adjective", "")
        audience_persona = agents_data.get("audience_persona", {})
        competitive_landscape = agents_data.get("competitive_landscape", {})

        product_details = project_doc.get("product_details", "")
        company_url = project_doc.get("company_url", "")
        product_url = project_doc.get("product_url") or "Not provided"

        # Validation
        logger.info(f"[{agent_key}] Validating required inputs.")
        missing_inputs = []
        if not brand_adjective: missing_inputs.append("brand_adjective")
        if not audience_persona: missing_inputs.append("audience_persona")
        if not competitive_landscape: missing_inputs.append("competitive_landscape")
        if not product_details: missing_inputs.append("product_details")
        if not raw_text: missing_inputs.append("company_research.raw_text")
        if not company_url: missing_inputs.append("company_url")
        
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
        
        whitespace_opportunity = competitive_landscape.get("whitespace_opportunity", "")

        prompt = f"""You are a senior brand strategist identifying the "Central Human Truth" — the deep emotional problem or desire this product speaks to.

You will use the provided Audience Persona and Competitive Landscape whitespace to inform this truth.

RULES:
1. The truth MUST be about the audience, not about the product.
2. It MUST be specific enough to be surprising. It must NOT echo something a competitor is already saying.
3. You MUST firmly use the whitespace opportunity from the competitive landscape as a negative filter (ensure the truth leans into the whitespace, not existing competitor territories).

EXAMPLES:
- Bad truth: "People want to feel beautiful."
- Good truth: "Women in their late 20s feel invisible when their skin stops reflecting the confidence they have spent years building."

INPUTS:

Brand Adjective:
{brand_adjective}

Audience Persona:
{json.dumps(audience_persona, indent=2)}

Competitive Landscape Whitespace Opportunity:
{whitespace_opportunity}

Full Competitive Landscape:
{json.dumps(competitive_landscape, indent=2)}

Product Details:
{product_details}

Product Webpage URL:
{product_url}
(If a product URL is provided, use it to research the specific product in depth.)

Company URL:
{company_url}

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
                "response_json_schema": CentralHumanTruthResult.model_json_schema(),
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

        # Separate reasoning from the main output
        reasoning = truth_data.pop("reasoning", "")

        # 4. Save to db (Strategy Collection)
        logger.info(f"[{agent_key}] Writing generated truth to strategy.agents.central_human_truth.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": {"agents.central_human_truth": truth_data}}
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
