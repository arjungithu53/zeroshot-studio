import os
import json
import time
import logging
from datetime import datetime, timezone
from bson import ObjectId
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
PIPELINE_COLLECTION  = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION  = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION  = os.getenv("COLLECTION_PROJECTS", "projects")

# Setup logger
logger = logging.getLogger("zeroshot.truth_conflict_platform")

class PlatformCandidate(BaseModel):
    platform: str = Field(description="The platform idea, a short declarative sentence")
    strategic_clarity: int = Field(description="Score 0-10")
    creative_potential: int = Field(description="Score 0-10")
    whitespace_score: int = Field(description="Score 0-10")
    total: int = Field(description="Sum of all scores")
    reasoning: str = Field(description="Explanation of why this platform candidate is strong or weak")

class TruthConflictPlatformResult(BaseModel):
    platform_candidates: list[PlatformCandidate]
    selected_platform: str = Field(description="The final selected platform")
    selected_index: int = Field(description="The index of the selected candidate")

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

async def run_truth_conflict_platform_agent(project_id: str, db):
    agent_key = "truth_conflict_platform"
    timestamp = datetime.now(timezone.utc)
    
    logger.info(f"[{agent_key}] Starting agent. project_id={project_id}")
    
    try:
        # 1. Fetch data from DB
        logger.info(f"[{agent_key}] Fetching strategy document.")
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": project_id})
        if not strategy_doc:
            raise ValueError(f"Strategy document for project {project_id} not found.")
            
        strategy_id = strategy_doc["_id"]

        # Extract fields
        agents_data = strategy_doc.get("agents", {})
        central_human_truth = agents_data.get("central_human_truth", {})
        truest_thing = agents_data.get("truest_thing", {})
        conflict_identification = agents_data.get("conflict_identification", {})

        # Validation
        logger.info(f"[{agent_key}] Validating required inputs.")
        missing_inputs = []
        if not central_human_truth: missing_inputs.append("central_human_truth")
        if not truest_thing: missing_inputs.append("truest_thing")
        if not conflict_identification: missing_inputs.append("conflict_identification")
        
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

        prompt = f"""You are an elite Brand Strategist. Your objective is to formulate a Campaign Platform that acts as the storytelling engine for all future creative executions.

A powerful Campaign Platform relies on the formula: Truth + Conflict = Platform. By combining the unvarnished, undeniable truth about a brand with the natural tension or conflict that arises from that truth, you create a specific narrative universe (a "World") with its own inherent rules (e.g., Truth: Crocs are ugly + Conflict: You won't get laid = Platform: "The World of Contraception").

Execute the following campaign platform workflow:

SYNTHESIS: Merge the Truest Thing and Central Human Truth to establish the unvarnished reality of the brand. Cross-reference this reality with the Conflict Identification to establish the core tension.

IDEATION: Generate exactly 3 distinct Campaign Platform candidates. For each candidate, define the specific narrative "World" it inhabits (e.g., "The World of Scandal," "The World of the Crazy Ex").

EVALUATION: Score each candidate strictly from 0 to 10 on three independent axes:

Strategic Clarity: Measure how perfectly the platform aligns with the synthesized truth and conflict.

Creative Potential: Measure the platform's capacity to generate dozens of diverse, engaging stories within its specific "World".

Ownable Whitespace: Measure how unique this narrative territory is compared to generic category norms.

SELECTION: Calculate the total score for each candidate and select the strongest platform index to drive the campaign.

CONSTRAINTS & RULES:

Ground the unvarnished truth strictly in the provided inputs. Embrace uncomfortable or raw truths if they are authentic to the data.

Frame the platform exclusively as a strategic narrative universe and internal guiding territory. Exclude all consumer-facing taglines, slogans, or copy.

Ensure the narrative "World" establishes clear, usable rules for story generation.

OUTPUT FORMAT:
Respond ONLY with valid JSON. Do not include markdown formatting or preamble. Use the exact schema below:
{{
"platform_candidates":[
{{
"index": 0,
"truth_and_conflict_synthesis": "<1 sentence summarizing the raw truth and resulting conflict>",
"narrative_world": "<Name of the specific narrative universe, e.g., 'The World of Scandal'>",
"platform_statement": "<Short declarative sentence defining the campaign territory>",
"strategic_clarity_score": <int 0-10>,
"creative_potential_score": <int 0-10>,
"ownable_whitespace_score": <int 0-10>,
"total_score": <int 0-30>
}},
{{
"index": 1,
"truth_and_conflict_synthesis": "<1 sentence summarizing the raw truth and resulting conflict>",
"narrative_world": "<Name of the specific narrative universe>",
"platform_statement": "<Short declarative sentence defining the campaign territory>",
"strategic_clarity_score": <int 0-10>,
"creative_potential_score": <int 0-10>,
"ownable_whitespace_score": <int 0-10>,
"total_score": <int 0-30>
}},
{{
"index": 2,
"truth_and_conflict_synthesis": "<1 sentence summarizing the raw truth and resulting conflict>",
"narrative_world": "<Name of the specific narrative universe>",
"platform_statement": "<Short declarative sentence defining the campaign territory>",
"strategic_clarity_score": <int 0-10>,
"creative_potential_score": <int 0-10>,
"ownable_whitespace_score": <int 0-10>,
"total_score": <int 0-30>
}}
],
"selected_index": <int 0-2>
}}

INPUT DATA:

CENTRAL HUMAN TRUTH:
{json.dumps(central_human_truth, indent=2)}

TRUEST THING:
{json.dumps(truest_thing, indent=2)}

CONFLICT IDENTIFICATION:
{json.dumps(conflict_identification, indent=2)}"""

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
                "response_json_schema": TruthConflictPlatformResult.model_json_schema(),
            }
        )
        api_duration = time.time() - start_time
        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")

        # 3. Parse output
        logger.info(f"[{agent_key}] Parsing and cleaning JSON response.")
        cleaned_json_str = clean_json_string(response.text)
        try:
            platform_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as je:
            raise ValueError(f"Failed to parse JSON from Gemini response: {je}. Raw response: {response.text}")

        # Extract reasoning to save to pipeline
        reasoning = ""
        selected_index = platform_data.get("selected_index", 0)
        candidates = platform_data.get("platform_candidates", [])
        
        if 0 <= selected_index < len(candidates):
            reasoning = candidates[selected_index].get("reasoning", "")
            
        selected_platform = platform_data.get("selected_platform", "")
        
        # 4. Save to db
        logger.info(f"[{agent_key}] Writing generated platform to strategy.agents.{agent_key}.")
        # Only saving the selected_platform to the strategy collection
        strategy_payload = {
            "selected_platform": selected_platform
        }
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": {f"agents.{agent_key}": strategy_payload}}
        )

        # 5. Log success to pipeline
        logger.info(f"[{agent_key}] Writing full data to pipeline document.")
        success_log = {
            "agent_key": agent_key,
            "status": "completed",
            "timestamp": timestamp,
            "platform_data": platform_data, # Store everything (including candidates, scores, reasoning, etc) in pipeline
            "reasoning": reasoning
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": success_log}},
            upsert=True
        )

        logger.info(f"[{agent_key}] Execution completed successfully.")
        
        return platform_data

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
