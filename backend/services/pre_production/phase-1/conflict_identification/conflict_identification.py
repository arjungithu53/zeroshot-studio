import os
import json
import time
import logging
from datetime import datetime, timezone
from bson import ObjectId
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
PIPELINE_COLLECTION  = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION  = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION  = os.getenv("COLLECTION_PROJECTS", "projects")

# Setup logger
logger = logging.getLogger("zeroshot.conflict_identification")

class ConflictIdentificationResult(BaseModel):
    enemy: str = Field(description="The identified enemy")
    conflict_statement: str = Field(description="[Audience] vs [Enemy] dramatic tension")
    enemy_type: Literal["feeling", "condition", "social_norm", "competitor", "old_behaviour", "other"] = Field(description="Type of enemy")
    reasoning: str = Field(description="Explanation of why this enemy and conflict were chosen")

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

async def run_conflict_identification_agent(project_id: str, db):
    agent_key = "conflict_identification"
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
        audience_persona = agents_data.get("audience_persona", {})
        visual_context_summary = strategy_doc.get("visual_context_summary", "")
        competitive_landscape = agents_data.get("competitive_landscape", {})

        product_details = project_doc.get("product_details", "")
        product_url = project_doc.get("product_url") or "Not provided"

        # Validation
        logger.info(f"[{agent_key}] Validating required inputs.")
        missing_inputs = []
        if not central_human_truth: missing_inputs.append("central_human_truth")
        if not truest_thing: missing_inputs.append("truest_thing")
        if not value_prop_and_offer: missing_inputs.append("value_prop_and_offer")
        if not audience_persona: missing_inputs.append("audience_persona")
        # if not competitive_landscape: missing_inputs.append("competitive_landscape")
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

        prompt = f"""Act as an elite Brand Strategist. Your objective is to define the core "Conflict" (the enemy) for a brand's narrative.

All stories run on conflict. Without a bad guy or a point of tension, a brand's messaging defaults to a boring "solution" that customers will ignore. Human brains are wired to heed warnings and engage with tension. The enemy does not need to be a rival company; it can be a feeling, a condition, a social norm, a bad habit, or an old way of doing things.

Execute the following brand conflict workflow:

BOUNDARY VERIFICATION: Analyze the competitive landscape to map the enemies and conflicts already claimed by rival brands.

TENSION IDENTIFICATION: Synthesize the Central Human Truth, the Truest Thing, and the Audience Persona to isolate the exact friction point in the customer's life. Identify the negative force that prevents the customer from reaching their desired state.

CONFLICT FRAMING: Define the dramatic tension using a strict "[Audience] vs [Enemy]" structure.

CONSTRAINTS & RULES:

Ground the enemy in a specific, lived frustration or fear of the target persona.

Restrict the selected enemy strictly to the available competitive whitespace. Exclude any conflict territories occupied by competitors.

Formulate the conflict as a dramatic, unresolved tension. Do not frame a "solution" as the conflict.

Target the emotional or behavioral root of the problem rather than an abstract, external corporate metric.

OUTPUT FORMAT:
Respond ONLY with valid JSON. Do not include markdown formatting or preamble. Use the exact schema below:
{{
"competitor_conflict_verification": "<1-2 sentences identifying the claimed enemies to avoid>",
"enemy_type": "<Identify the category of the enemy, e.g., Social Norm, Internal Feeling, Outdated System, Bad Habit>",
"enemy": "<The specific name or description of the enemy>",
"conflict_statement": "<[Specific Audience] vs [Specific Enemy]>",
"reasoning": "<2-3 sentences justifying the choice based on the persona's resentment and the competitive whitespace>"
}}

INPUT DATA:

CENTRAL HUMAN TRUTH:
{json.dumps(central_human_truth, indent=2)}

TRUEST THING:
{json.dumps(truest_thing, indent=2)}

VALUE PROP & OFFER:
{json.dumps(value_prop_and_offer, indent=2)}

AUDIENCE PERSONA:
{json.dumps(audience_persona, indent=2)}

COMPETITIVE LANDSCAPE:
{json.dumps(competitive_landscape, indent=2)}

VISUAL CONTEXT SUMMARY:
{visual_context_summary}

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
                "response_json_schema": ConflictIdentificationResult.model_json_schema(),
            }
        )
        api_duration = time.time() - start_time
        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")

        # 3. Parse output
        logger.info(f"[{agent_key}] Parsing and cleaning JSON response.")
        cleaned_json_str = clean_json_string(response.text)
        try:
            conflict_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as je:
            raise ValueError(f"Failed to parse JSON from Gemini response: {je}. Raw response: {response.text}")

        # Extract reasoning and data to save to strategy
        reasoning = conflict_data.pop("reasoning", "")
        
        # 4. Save to db
        logger.info(f"[{agent_key}] Writing generated conflict to strategy.agents.conflict_identification.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": {f"agents.{agent_key}": conflict_data}}
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
        
        # Add reasoning back to the returned dict for caller convenience
        conflict_data["reasoning"] = reasoning
        return conflict_data

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
