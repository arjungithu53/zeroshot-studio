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
logger = logging.getLogger("zeroshot.truest_thing")

class Candidate(BaseModel):
    statement: str = Field(description="The functional to deeply emotional statement")
    specificity: int = Field(description="Score from 0-10: could this apply to a competitor?")
    defensibility: int = Field(description="Score from 0-10: can the product actually back this up?")
    resonance: int = Field(description="Score from 0-10: does it trigger a felt response, not just intellectual agreement?")
    total: int = Field(description="Sum of specificity, defensibility, and resonance scores")

class TruestThingResult(BaseModel):
    candidates: List[Candidate] = Field(description="Exactly 5 candidate statements evaluated")
    truest_thing: str = Field(description="The highest-scoring candidate statement")
    selected_index: int = Field(description="The index of the selected candidate statement (0-4)")

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

async def run_truest_thing_agent(project_id: str, db):
    agent_key = "truest_thing"
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
        brand_adjective = agents_data.get("brand_adjective", {})
        central_human_truth = agents_data.get("central_human_truth", {})
        value_prop_and_offer = agents_data.get("value_prop_and_offer", {})

        product_details = project_doc.get("product_details", "")
        product_url = project_doc.get("product_url") or "Not provided"

        # Validation
        logger.info(f"[{agent_key}] Validating required inputs.")
        missing_inputs = []
        if not brand_adjective: missing_inputs.append("brand_adjective")
        if not central_human_truth: missing_inputs.append("central_human_truth")
        if not value_prop_and_offer: missing_inputs.append("value_prop_and_offer")
        if not product_details: missing_inputs.append("product_details")
        
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

        prompt = f"""You are an elite Brand Strategist. Your objective is to define the "Truest Thing" a brand can say.

Customers ignore measurable product facts (e.g., "milk builds strong bones"). They connect with the undeniable, universally true reality of their own behavior (e.g., "milk goes with things"). You must transcend corporate claims and identify the foundational truth that all future messaging will strictly adhere to.

Execute the following brand truth formulation workflow:

SYNTHESIS: Merge the Central Human Truth (the customer's emotional reality) with the Value Prop & Offer (the product's functional role), filtered strictly through the lens of the Brand Adjective.

IDEATION: Generate exactly 5 distinct candidate statements representing the truest things the brand can say. Range these from functional realities to deep emotional anchors.

EVALUATION: Score each candidate strictly from 0 to 10 on three independent axes:

Specificity: Measure how uniquely this applies to the target brand versus its competitors (10 = Highly specific to this brand, 0 = Generic category claim).

Defensibility: Measure the product's ability to back this up with empirical proof (10 = Absolute proof exists within product details, 0 = Baseless marketing claim).

Resonance: Measure the capacity to trigger a felt, emotional response rather than mere intellectual agreement (10 = Massive emotional resonance, 0 = Dry clinical fact).

SELECTION: Calculate the total score (Specificity + Defensibility + Resonance) for each candidate. Select the highest-scoring candidate to serve as the foundational truth.

CONSTRAINTS & RULES:

Ground all candidate statements strictly in the provided input data.

Exclude measurable company facts, statistics, and feature lists from the candidate truths.

Format the final selection as a foundational internal messaging truth, excluding all catchy external taglines, slogans, or copywritten hooks.

Base all resonance scoring on authentic human behavior and ruthlessly penalize corporate posturing or phony marketing ideas.

OUTPUT FORMAT:
Respond ONLY with valid JSON. Do not include markdown formatting, code blocks, or preamble. Use the exact schema below:
{{
"candidate_evaluation":[
{{
"index": 0,
"candidate_statement": "<statement>",
"specificity_score": <int 0-10>,
"defensibility_score": <int 0-10>,
"resonance_score": <int 0-10>,
"total_score": <int 0-30>
}},
{{
"index": 1,
"candidate_statement": "<statement>",
"specificity_score": <int 0-10>,
"defensibility_score": <int 0-10>,
"resonance_score": <int 0-10>,
"total_score": <int 0-30>
}},
{{
"index": 2,
"candidate_statement": "<statement>",
"specificity_score": <int 0-10>,
"defensibility_score": <int 0-10>,
"resonance_score": <int 0-10>,
"total_score": <int 0-30>
}},
{{
"index": 3,
"candidate_statement": "<statement>",
"specificity_score": <int 0-10>,
"defensibility_score": <int 0-10>,
"resonance_score": <int 0-10>,
"total_score": <int 0-30>
}},
{{
"index": 4,
"candidate_statement": "<statement>",
"specificity_score": <int 0-10>,
"defensibility_score": <int 0-10>,
"resonance_score": <int 0-10>,
"total_score": <int 0-30>
}}
],
"selected_index": <int 0-4>,
"truest_thing": "<exact string of the highest-scoring statement>"
}}

INPUT DATA:

BRAND ADJECTIVE:
{json.dumps(brand_adjective, indent=2)}

CENTRAL HUMAN TRUTH (AGENT 4):
{json.dumps(central_human_truth, indent=2)}

VALUE PROP & OFFER (AGENT 5):
{json.dumps(value_prop_and_offer, indent=2)}

PRODUCT DETAILS:
{product_details}

PRODUCT WEBPAGE URL:
{product_url}
(If a product URL is provided, use it to research the specific product in depth.)"""

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
                "response_json_schema": TruestThingResult.model_json_schema(),
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

        # 4. Save to db
        logger.info(f"[{agent_key}] Writing generated truest thing to strategy.agents.truest_thing.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": {f"agents.{agent_key}": truth_data.get("truest_thing", "")}}
        )

        # 5. Log success to pipeline
        logger.info(f"[{agent_key}] Writing success log to pipeline document.")
        success_log = {
            "agent_key": agent_key,
            "status": "completed",
            "candidates": truth_data.get("candidates", []),
            "selected_index": truth_data.get("selected_index"),
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
