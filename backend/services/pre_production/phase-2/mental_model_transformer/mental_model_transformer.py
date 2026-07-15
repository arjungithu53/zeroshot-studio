import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from bson import ObjectId
from google import genai

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Load environment variables
load_dotenv()

# Global Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")

# Pydantic Output Models
class LensProcessingItem(BaseModel):
    lens: str = Field(description="The divergence lens applied")
    seed: str = Field(description="The 2-sentence concept seed generated for this lens")

class MentalModelTransformerResult(BaseModel):
    reasoning: str = Field(description="Pipeline reasoning for divergence generation")
    lens_processing_table: List[LensProcessingItem] = Field(description="The table of 7 generated seeds labeled by lens")
    seed_list: List[str] = Field(description="Flat list of the 7 generated seeds")
    status: Optional[str] = Field("completed", description="Status of the agent execution")

def _clean_json_string(text: str) -> str:
    """Removes markdown code blocks around JSON if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()
    return text

async def run_mental_model_transformer(project_id: str, db) -> MentalModelTransformerResult:
    """
    Agent 18: mental_model_transformer
    Processes human_truth + truest_thing through 7 divergence lenses to create unique concept seeds.
    """
    logger.info(f"Initializing Agent 18 for project_id={project_id}...")
    
    start_time = time.time()
    
    try:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing.")
            
        logger.info(f"Agent 18: Fetching data for project_id={project_id}")
        
        # 1. Fetch from STRATEGY_COLLECTION
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        if not strategy_doc:
            raise ValueError(f"No strategy document found for project {project_id}")
            
        strategy_agents = strategy_doc.get("agents", {})
        
        def _safe_get(agent_data, key):
            if isinstance(agent_data, dict):
                return agent_data.get(key, "")
            elif isinstance(agent_data, str):
                return agent_data
            return ""
            
        human_truth = _safe_get(strategy_agents.get("central_human_truth", {}), "human_truth")
        truest_thing = _safe_get(strategy_agents.get("truest_thing", {}), "truest_thing")
        campaign_platform = _safe_get(strategy_agents.get("truth_conflict_platform", {}), "selected_platform")
        enemy = _safe_get(strategy_agents.get("conflict_identification", {}), "core_enemy")
        brand_adjective = _safe_get(strategy_agents.get("brand_adjective", {}), "brand_adjective")
        
        logger.info(f"Agent 18: Extracted strategy inputs (human_truth, truest_thing, campaign_platform, enemy, brand_adjective).")

        # Construct prompt based on Agent 18 logic
        prompt = f"""
        You are Agent 18: mental_model_transformer.
        
        Your task is to take a core truth and pass it through 7 distinct concepting lenses.
        The goal is maximum divergence in how the same truth is expressed — not surface copy variation, but genuine structural and visual divergence across the concept portfolio.
        
        INPUTS:
        - human_truth: {json.dumps(human_truth)}
        - truest_thing: {json.dumps(truest_thing)}
        - campaign_platform: {json.dumps(campaign_platform)}
        - enemy: {json.dumps(enemy)}
        - brand_adjective: {json.dumps(brand_adjective)}
        
        PROMPT LOGIC INSTRUCTIONS:
        
        The main message is the combination of human_truth and truest_thing. Process it through the following seven lenses:
        
        1. Before and after: What is the most specific, visceral version of the before state and the after state — same environment, same woman, but something has shifted in her?
        2. Passage of time: What if the concept was structured as a compressed time sequence — showing the ritual accumulating meaning over weeks, months, or years of the same hostile environment?
        3. Change perspective: What if the ad was told from the perspective of the enemy and showed its failure?
        4. Abstract: What if the product's benefit was shown as pure color and sensation with no literal product demonstration at all?
        5. Simulate: What if the concept recreated the exact sensory experience of the enemy, and placed the product as the only warm thing in the frame?
        6. Combine: What if the product's origin and the modern context were shown simultaneously — split-screen, overlapping timelines, or visual morphing?
        7. Dissect: What if we broke the product into its ingredients and showed each one's origin story as a rapid micro-sequence?
        
        For each lens, produce exactly one 2-sentence concept seed describing the execution direction.
        
        OUTPUT FORMAT:
        Return a strict JSON object matching the requested schema exactly.
        """

        invoke_start = time.time()
        logger.info(f"Agent 18: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 18: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": MentalModelTransformerResult.model_json_schema(),
                "temperature": 1.0,
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 18: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 18: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 18: Successfully parsed JSON response.")

        result = MentalModelTransformerResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 18: Successfully validated structured output with Pydantic.")
        
        pipeline_log = {
            "agent_id": "mental_model_transformer",
            "agent_name": "Agent 18 - mental_model_transformer",
            "execution_duration_sec": round(time.time() - start_time, 2),
            "api_duration_sec": round(api_duration, 2),
            "timestamp": time.time(),
            "data": result.model_dump()
        }
        
        logger.info(f"Agent 18: Updating PIPELINE collection...")
        # Note: ONLY pushing to pipeline_logs. No ideation write as per requirements.
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )
        logger.info(f"Agent 18: Database update verified.")
        
        total_duration = time.time() - start_time
        logger.info(f"Agent 18: Completed successfully in {total_duration:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Agent 18 execution failed: {str(e)}", exc_info=True)
        raise