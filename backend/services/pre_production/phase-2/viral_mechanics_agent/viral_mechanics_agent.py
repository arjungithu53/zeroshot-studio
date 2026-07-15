import os
import time
import json
import logging
from typing import Any
from bson import ObjectId
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai

load_dotenv()

# Global configuration variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")

logger = logging.getLogger("zeroshot.phase2.viral_mechanics")

class ViralMechanicsResult(BaseModel):
    reasoning: str = Field(description="General reasoning for the mechanics evaluation.")
    social_currency_eval: str = Field(description="Evaluation of the social currency mechanic.")
    practical_value_eval: str = Field(description="Evaluation of the practical value mechanic.")
    trigger_eval: str = Field(description="Evaluation of the trigger mechanic.")
    selected_mechanic: str = Field(description="The single strongest selected virality mechanic.")
    virality_directive: str = Field(description="A 2-sentence virality directive specifying which mechanic to embed and when it activates structurally.")

def _clean_json_string(json_str: str) -> str:
    """Helper to clean markdown formatting from JSON string."""
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    return json_str.strip()

async def run_viral_mechanics_agent(project_id: str, db: Any) -> dict:
    logger.info(f"Initializing Agent 17 (Viral Mechanics Agent) for project_id={project_id}...")
    
    logger.info(f"Agent 17: Fetching data for project_id={project_id}")
    
    try:
        project_obj_id = ObjectId(project_id)
    except Exception as e:
        logger.error(f"Agent 17: Invalid project_id format. Error: {str(e)}")
        raise ValueError(f"Invalid project_id {project_id}")

    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": project_obj_id})
    if not project_doc:
        logger.error(f"Agent 17: Project {project_id} not found in {PROJECTS_COLLECTION}.")
        raise ValueError(f"Project {project_id} not found.")

    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}

    # Extract required inputs
    agents_data = strategy_doc.get("agents", {})
    persona = agents_data.get("audience_persona", {})
    brand_adjective = agents_data.get("brand_adjective", {})
    value_prop_data = agents_data.get("value_prop_and_offer", {})
    offer_hook = value_prop_data.get("offer_hook", "")
    human_truth = agents_data.get("central_human_truth", "")
    campaign_platform = agents_data.get("truth_conflict_platform", "")
    
    platform_rules = ideation_doc.get("platform_rules", {})

    logger.info("Agent 17: Extracted persona, offer_hook, human_truth, campaign_platform, and platform_rules from database.")

    prompt = f"""
You are Viral Mechanics Agent.
Your Purpose: Grounded in the Contagious framework, evaluate three virality mechanics against the brief: social currency, practical value, and trigger. Select the strongest lever for this persona and write a 2-sentence virality directive specifying which mechanic to embed in every concept and at which structural beat it activates.

INPUT DATA:
Persona: {json.dumps(persona, indent=2)}
Brand Adjective: {json.dumps(brand_adjective, indent=2)}
Offer Hook: {offer_hook}
Human Truth: {json.dumps(human_truth, indent=2)}
Campaign Platform: {json.dumps(campaign_platform, indent=2)}
Platform Rules: {json.dumps(platform_rules, indent=2)}

PROMPT LOGIC:
The source framework identifies three virality mechanics most applicable to social video advertising: social currency, practical value, and triggers. Evaluate all three for this specific brief and select the virality lever best matched to the persona's psychology.

1. Social Currency evaluation:
People share what makes them look good — smart, in-the-know, culturally aware. Using the Brand Adjective and Persona data: does the positioning and sourcing contain a shareable fact or visual that would make the persona feel elevated for knowing it? Generate one social-currency-based virality hook if this mechanism applies.

2. Practical Value evaluation:
People share what helps their friends. Using the Offer Hook: does the offer create a natural reason to tag friends or split the purchase? How can the offer be narratively positioned as an act of giving rather than selling? Generate one practical-value-based virality hook if this mechanism applies.

3. Trigger evaluation:
People share what connects to something already in their mental environment. Using the Persona's daily life: is there a recurring sensory trigger in the persona's daily life this campaign can attach to, so that every time she encounters that stimulus she thinks of this product? Generate one trigger-based virality hook.

Select the single strongest virality lever and write a 2-sentence virality directive for Agent 20 specifying: (1) which mechanic to embed in every concept, and (2) at what structural moment in the narrative it should activate.

Return your response formatted strictly as a JSON object matching the requested schema.
"""

    invoke_start = time.time()
    logger.info(f"Agent 17: Preparing to call Gemini model={GEMINI_MODEL}...")
    
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info("Agent 17: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ViralMechanicsResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 17: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 17: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info("Agent 17: Successfully parsed JSON response.")

        result = ViralMechanicsResult(**parsed_data)
        logger.info("Agent 17: Successfully validated structured output with Pydantic.")
        
    except Exception as e:
        logger.error(f"Agent 17: Error during Gemini inference or JSON parsing: {str(e)}")
        raise

    try:
        logger.info("Agent 17: Updating PIPELINE_COLLECTION...")
        pipeline_log = {
            "agent_id": 17,
            "agent_name": "viral_mechanics_agent",
            "execution_time_sec": round(time.time() - invoke_start, 2),
            "timestamp": time.time(),
            "status": "completed",
            "output": parsed_data
        }

        upd = await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )
        if upd.matched_count == 0 and upd.upserted_id is None:
            logger.error("Agent 17: Failed to update or upsert standard pipeline log.")
        else:
            logger.info("Agent 17: Successfully updated PIPELINE_COLLECTION.")
            
    except Exception as e:
        logger.error(f"Agent 17: Error writing to DB: {str(e)}")
        raise

    total_duration = time.time() - invoke_start
    logger.info(f"Agent 17: Execution completed in {total_duration:.2f}s for project_id={project_id}")

    return parsed_data
