import json
import logging
import os
import time
from typing import List

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.intergalactic_thinking_agent")

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")

# ── Pydantic Models for Structured Output ──

class DistantGalaxy(BaseModel):
    galaxy_name: str = Field(description="Name of the distant galaxy (unrelated domain)")
    data_points: List[str] = Field(description="List of 3 data points from this distant galaxy")

class ConnectionFound(BaseModel):
    home_point: str = Field(description="Data point from the home galaxy")
    galaxy_point: str = Field(description="Data point from the distant galaxy")
    concept_direction: str = Field(description="The conceptually rich connection direction created")

class IntergalacticThinkingResult(BaseModel):
    reasoning: str = Field(description="Explanation of the extraction and connection methodology")
    home_galaxy: List[str] = Field(description="List of 30 facts about brand and category")
    distant_galaxies: List[DistantGalaxy] = Field(description="Three distant galaxies with rich data points")
    connections_found: List[ConnectionFound] = Field(description="The 2 most unexpected, creatively productive connections")
    seed_list: List[str] = Field(description="Write these as 2 intergalactic concept seeds with a 3-sentence execution direction each.")
    status: str = Field(default="completed", description="Execution status")

def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

async def run_intergalactic_thinking_agent(project_id: str, db) -> IntergalacticThinkingResult:
    """
    Agent 19: Intergalactic Thinking Agent
    Builds a home galaxy (30 data points), builds 3 distant galaxies (unrelated domains),
    connects data points across galaxies to discover 2 high-novelty concept directions
    that standard briefs would not produce.
    """
    agent_key = "intergalactic_thinking_agent"
    logger.info(f"[{agent_key}] Starting agent execution | project_id={project_id}")
    start_time = time.time()

    try:
        # Fetch project data (using ObjectId for the projects collection)
        logger.info(f"[{agent_key}] Fetching projects document for project_id={project_id}")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            logger.error(f"[{agent_key}] Project not found | project_id={project_id}")
            raise ValueError(f"Project '{project_id}' not found")
            
        product_details = project_doc.get("product_details", "No product details provided.")
        product_url = project_doc.get("product_url") or "Not provided"
        
        # Fetch strategy data
        logger.info(f"[{agent_key}] Fetching strategy document for project_id={project_id}")
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        if not strategy_doc:
            logger.error(f"[{agent_key}] Strategy document not found | project_id={project_id}")
            raise ValueError(f"Strategy for project '{project_id}' not found")

        # Extract strategy fields safely
        agents_data = strategy_doc.get("agents", {})
        
        brand_adjective = agents_data.get("brand_adjective", "N/A")
        human_truth = agents_data.get("central_human_truth", {}).get("human_truth", "N/A")
        if human_truth == "N/A": 
             human_truth = agents_data.get("human_truth", "N/A")
             
        # "value_prop_and_offer" extraction might be buried or top level in agents
        value_prop_and_offer = agents_data.get("value_prop_and_offer", "N/A")
        
        # Conflict / Enemy
        enemy = agents_data.get("conflict_identification", {}).get("enemy", "N/A")
        
        campaign_platform = strategy_doc.get("truth_conflict_platform", {}).get("campaign_platform", "N/A")
        if campaign_platform == "N/A":
            campaign_platform = agents_data.get("campaign_platform", "N/A")

        # Top level strategy fields
        company_research_raw = strategy_doc.get("company_research", {}).get("raw_text", "N/A")
        visual_context_summary = strategy_doc.get("visual_context_summary", "N/A")

        logger.info(f"[{agent_key}] Data extraction complete. Extracted elements correctly.")

        prompt = f'''
        You are the Intergalactic Thinking Agent. 
        Your purpose is to build a "home galaxy" of facts about the brand/category, then build "distant galaxies" 
        from unrelated domains with rich data points, and finally connect data points across galaxies 
        to discover unexpected concept directions.

        ## Source Framework Context
        Step 1 — Populate the home galaxy. List 30 data points about this brand and category, drawn from the input data. Include ingredients, certifications, price point, brand tone, audience context, product texture, usage ritual, and competitive positioning.
        Step 2 — Build three distant galaxies far from the category. Use domains with rich data points (e.g., Ancient astronomy, The spice trade, Textile weaving traditions).
        Step 3 — Pick 3 data points from each distant galaxy and connect them to the home galaxy's creative problem. The creative problem is derived from the human_truth and enemy: how do we solve the user's conflict using this product platform.
        Step 4 — Extract the 2 most unexpected, creatively productive connections. Write these as 2 intergalactic concept seeds with a 3-sentence execution direction each.

        ## Input Data
        **brand_adjective**: "{brand_adjective}"
        **human_truth**: "{human_truth}"
        **enemy**: "{enemy}"
        **campaign_platform**: "{campaign_platform}"
        **value_prop_and_offer**: "{value_prop_and_offer}"
        
        **product_details**: "{product_details}"
        **product_url**: "{product_url}"
        (If a product URL is provided, use it to research the specific product in depth when building the home galaxy.)
        **company_research_raw**: "{company_research_raw}"
        **visual_context_summary**: "{visual_context_summary}"
        
        ## Output
        Provide the response purely in JSON format matching the schema rules, extracting reasoning, the home galaxy list, the distant galaxies list, connections found, and lastly the two concept seeds.
        '''

        invoke_start = time.time()
        logger.info(f"Agent [{agent_key}]: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent [{agent_key}]: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": IntergalacticThinkingResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent [{agent_key}]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent [{agent_key}]: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent [{agent_key}]: Successfully parsed JSON response.")

        result = IntergalacticThinkingResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent [{agent_key}]: Successfully validated structured output with Pydantic.")

        logger.info(f"Agent [{agent_key}]: Updating IDEATION and PIPELINE collections...")

        # Store seeds in ideation for reliable typed retrieval by concept_generator
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "intergalactic_seeds": result.seed_list,
                "status.intergalactic_thinking_agent": "completed",
            }},
            upsert=True,
        )

        pipeline_log = {
            "agent_id": "agent_19",
            "agent_name": "intergalactic_thinking_agent",
            "execution_time_seconds": round(time.time() - start_time, 2),
            "timestamp": time.time(),
            "reasoning": result.reasoning,
            "home_galaxy": result.home_galaxy,
            "distant_galaxies": [dg.model_dump() for dg in result.distant_galaxies],
            "connections_found": [cf.model_dump() for cf in result.connections_found],
            "seed_list": result.seed_list,
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )

        total_duration = time.time() - start_time
        logger.info(f"[{agent_key}] Completed successfully in {total_duration:.2f}s | project_id={project_id}")

        return result

    except Exception as e:
        logger.error(f"[{agent_key}] Error running agent: {str(e)}", exc_info=True)
        raise e
