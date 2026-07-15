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
logger = logging.getLogger("zeroshot.strategy_models")

class VennModel(BaseModel):
    brand_can_say: List[str] = Field(description="What the brand can say")
    audience_cares_about: List[str] = Field(description="What the audience cares about")
    competitor_gap: List[str] = Field(description="What competitors are NOT saying")
    strategic_sweet_spot: str = Field(description="The intersection of the three sets")

class CreativeBrief(BaseModel):
    target_audience: str = Field(description="The target audience for the brief")
    single_minded_proposition: str = Field(description="The one single minded proposition informed by campaign platform and offer hook")
    support_points: List[str] = Field(description="Maximum of 3 support points", max_length=3)
    tone_of_voice: str = Field(description="Tone of voice")
    mandatories: List[str] = Field(description="Must reflect any stated constraints from brand guidelines")

class StrategyModelsResult(BaseModel):
    venn_model: VennModel
    creative_brief: CreativeBrief

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

async def run_strategy_models_agent(project_id: str, db):
    agent_key = "strategy_models"
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
        
        # Specific previous agent outputs for stronger context
        audience_persona = agents_data.get("audience_persona", {})
        competitive_landscape = agents_data.get("competitive_landscape", {})
        truth_conflict_platform = agents_data.get("truth_conflict_platform", {})
        value_prop_and_offer = agents_data.get("value_prop_and_offer", {})
        
        # Adding these missing ones which are directly relevant to this agent's task:
        brand_adjective = agents_data.get("brand_adjective", {}) # Essential for Tone of Voice
        truest_thing = agents_data.get("truest_thing", {}) # Essential for Brand Strengths
        
        # Pull required project parameters
        product_details = project_doc.get("product_details", "")
        target_audience = project_doc.get("target_audience", "")
        price_and_offer = project_doc.get("price_and_offer", "")
        brand_guidelines = project_doc.get("brand_guidelines", "")

        # 2. Call Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")

        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"}
        )

        prompt = f"""You are an elite Brand Strategist. Your objective is to synthesize all previous brand, customer, and competitor insights into an actionable creative strategy.

You will utilize two specific frameworks to achieve this: a three-circle Venn Diagram (Model A) to isolate the strategic sweet spot, and a Simplified Creative Brief (Model B) to guide the execution team.

Execute the following creative strategy workflow:

VENN DIAGRAM ANALYSIS (MODEL A):

Define Brand Strengths: Analyze the product details to identify the core capabilities and offerings of the brand.

Define Customer Desires: Analyze the target audience data to identify their exact needs and emotional drivers.

Define Competitor Focus: Analyze previous agent outputs to identify the specific messaging territories currently occupied by the competition.

Extract the Sweet Spot: Isolate the exact intersection answering the question: "What do we have that customers want and the competition isn't giving them?"

BRIEF SYNTHESIS (MODEL B):

Target Audience: Define exactly who the campaign is talking to and their core insight.

Single-Minded Proposition (SMP): Formulate the single most important message the audience needs to hear. You must synthesize this by merging the campaign platform and the offer hook identified in previous agent outputs.

Support Points: Identify the foundational reasons the audience should believe the SMP.

Tone of Voice: Define the specific emotional register the messaging should adopt.

Mandatories: Extract any explicit constraints, rules, or required elements from the brand guidelines.

CONSTRAINTS & RULES:

Ground all Venn Diagram analysis strictly in the provided input data and previous agent outputs.

Formulate the Single-Minded Proposition strictly using the Campaign Platform (Agent 9) and the Offer Hook (Agent 5) found in the previous agent data.

Limit the Support Points in Model B to a maximum of 3 distinct, measurable, or factual product details.

Extract Mandatories strictly from the provided Brand Guidelines. If no guidelines exist, explicitly output "None".

Format the output strictly as a JSON object utilizing the schema below.

OUTPUT FORMAT:
Respond ONLY with valid JSON. Do not include markdown formatting or preamble. Use the exact schema below:
{{
"model_a_venn_diagram": {{
"brand_strengths": "<1-2 sentences defining what the brand offers>",
"customer_desires": "<1-2 sentences defining what the audience wants>",
"competitor_focus": "<1-2 sentences defining what the competition is doing>",
"strategic_sweet_spot": "<1 sentence defining the exact intersection of what the brand has, the customer wants, and the competition ignores>"
}},
"model_b_creative_brief": {{
"target_audience": "<1 sentence defining the audience and their core desire>",
"single_minded_proposition": "<The one most important message, combining the campaign platform and offer hook>",
"support_points":[
"<support point 1>",
"<support point 2>",
"<support point 3>"
],
"tone_of_voice": "<1-2 words defining the emotional register>",
"mandatories": "<List of required elements or constraints from brand guidelines>"
}}
}}

INPUT DATA:

PRODUCT DETAILS:
{product_details}

TARGET AUDIENCE:
{target_audience}

PRICE AND OFFER:
{price_and_offer}

BRAND GUIDELINES:
{brand_guidelines}

AUDIENCE PERSONA (AGENT OUTPUT):
{json.dumps(audience_persona, indent=2)}

COMPETITIVE LANDSCAPE (AGENT OUTPUT):
{json.dumps(competitive_landscape, indent=2)}

CAMPAIGN PLATFORM / TRUTH CONFLICT (AGENT 9 OUTPUT):
{json.dumps(truth_conflict_platform, indent=2)}

OFFER HOOK / VALUE PROP (AGENT 5 OUTPUT):
{json.dumps(value_prop_and_offer, indent=2)}

BRAND ADJECTIVE (FOR TONE OF VOICE):
{json.dumps(brand_adjective, indent=2)}

TRUEST THING (FOR BRAND STRENGTHS):
{json.dumps(truest_thing, indent=2)}"""

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
                "response_json_schema": StrategyModelsResult.model_json_schema(),
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

        # 4. Save to db
        logger.info(f"[{agent_key}] Writing generated strategy models to strategy.agents.strategy_models.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": {f"agents.{agent_key}": result_data}}
        )

        # 5. Log success to pipeline
        logger.info(f"[{agent_key}] Writing success log to pipeline document.")
        success_log = {
            "agent_key": agent_key,
            "status": "completed",
            "timestamp": timestamp,
            "data": result_data
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": success_log}},
            upsert=True
        )

        logger.info(f"[{agent_key}] Execution completed successfully.")
        return result_data

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
        raise e
