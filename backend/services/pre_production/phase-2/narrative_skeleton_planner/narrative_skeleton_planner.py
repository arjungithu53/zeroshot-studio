import os
import json
import time
import logging
import traceback
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
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")

# Pydantic Output Models
class PerConceptSkeletonVariant(BaseModel):
    variant_id: str = Field(description="Variant Identifier (e.g., V1, V2...)")
    skeleton_mutation: str = Field(description="The structural mutation applied to the skeleton")
    rationale: str = Field(description="Why this mutation creates useful structural diversity")

class VariantGenerationLogEntry(BaseModel):
    variant_id: str = Field(description="Variant Identifier")
    mutation_type: str = Field(description="Type of mutation applied")
    expected_effect: str = Field(description="Expected narrative effect of the mutation")

class Agent13Result(BaseModel):
    archetype_alignment_check: str = Field(description="Verification of skeleton variants against archetype")
    per_concept_skeleton_variants: List[PerConceptSkeletonVariant] = Field(description="Skeleton variant per concept slot (V1-V6)")
    reasoning: str = Field(description="Reasoning log for pipeline")
    variant_generation_log: List[VariantGenerationLogEntry] = Field(description="Detailed generation log for pipeline")
    status: Optional[str] = Field("completed")

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

async def run_narrative_skeleton_planner(project_id: str, db) -> Dict[str, Any]:
    """
    Agent 13: narrative_skeleton_planner
    Plans how a master narrative skeleton is mutated across 6 concepts.
    """
    logger.info(f"Initializing Agent 13 for project_id={project_id}...")
    
    start_time = time.time()
    
    try:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing.")
            
        logger.info(f"Agent 13: Fetching data for project_id={project_id}")
        
        # 1. Fetch from IDEATION_COLLECTION
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"No ideation document found for project {project_id}")
            
        narrative_skeleton = ideation_doc.get("narrative_skeleton", {})
        priority_directives = ideation_doc.get("priority_directives", {})

        # Defensive guard: narrative planning is only valid for narrative-group formats.
        format_group = ideation_doc.get("format_group", "N")
        if format_group == "V":
            logger.info(f"Agent 13: Skipping — format_group is V (visual formats use visual_structure_agent) | project_id={project_id}")
            return {"success": True, "message": "narrative_skeleton_planner skipped — visual format group", "data": {"status": "skipped"}}

        logger.info(f"Agent 13: Extracted narrative_skeleton and priority_directives.")

        # 2. Fetch from STRATEGY_COLLECTION
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        campaign_platform = ""
        human_truth = ""
        if strategy_doc:
            strategy_agents = strategy_doc.get("agents", {})
            campaign_platform = strategy_agents.get("truth_conflict_platform", {}).get("selected_platform", "")
            human_truth = strategy_agents.get("central_human_truth", {}).get("human_truth", "")

        logger.info(f"Agent 13: Extracted campaign_platform and human_truth.")

        # Construct prompt based on Agent 13 logic
        prompt = f"""
        You are Agent 13: narrative_skeleton_planner.
        
        Your task is to receive a master narrative skeleton and plan how it is mutated across 6 distinct concepts.
        The goal is structural variation across the concept portfolio while preserving shared narrative grammar, preventing all 6 concepts from following the exact same sequence.
        
        INPUTS:
        - narrative_skeleton: {json.dumps(narrative_skeleton, indent=2)}
        - priority_directives: {json.dumps(priority_directives, indent=2)}
        - campaign_platform: {campaign_platform}
        - human_truth: {human_truth}
        
        PROMPT LOGIC INSTRUCTIONS:
        
        Step 1 - Identify the mutation axes available in the master skeleton.
        Determine which structural parameters can be varied without breaking emotional logic: 
        beat order (swapping non-dependent beats), beat weight (compressing into a transition), beat type (tension expressed as humor), beat count (3 beats vs 5 beats).
        
        Step 2 - Generate one skeleton variant per concept slot (6 variants total, labelled V1-V6).
        Each variant must specify the mutation applied and the rationale. 
        RULES:
        - No two variants can share the same mutation type on the same beat position.
        - At least one variant must compress the skeleton for a PITCH-category concept.
        - At least one must expand it for a PLUNGE-category concept.
        
        Step 3 - Run an archetype alignment check.
        Verify all 6 skeleton variants remain compatible with the narrative archetype (using platform and truth as proxies).
        Flag any variant incompatible with Ritual or Rebellion archetypes and note the incompatibility.
        
        OUTPUT FORMAT:
        Return a strict JSON object matching the requested schema exactly.
        """

        invoke_start = time.time()
        logger.info(f"Agent 13: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        genai_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 13: Gemini Client instantiated. Sending prompt...")
        
        response = await genai_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": Agent13Result.model_json_schema(),
                "temperature": 2.0,
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 13: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 13: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 13: Successfully parsed JSON response.")

        result = Agent13Result(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 13: Successfully validated structured output with Pydantic.")

        # DB Updates
        logger.info(f"Agent 13: Updating IDEATION collection...")
        
        ideation_update = {
            "narrative_plan": {
                "archetype_alignment_check": result.archetype_alignment_check,
                "per_concept_skeleton_variants": [v.model_dump() for v in result.per_concept_skeleton_variants]
            },
            "agents.narrative_skeleton_planner.status": "completed",
            "agents.narrative_skeleton_planner.updated_at": time.time()
        }
        
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": ideation_update},
            upsert=True
        )

        logger.info(f"Agent 13: Updating PIPELINE collection...")
        pipeline_log = {
            "agent_name": "narrative_skeleton_planner",
            "project_id": str(project_id),
            "timestamp": time.time(),
            "execution_time_seconds": time.time() - start_time,
            "reasoning": result.reasoning,
            "variant_generation_log": [v.model_dump() for v in result.variant_generation_log],
            "status": "success"
        }
        
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        total_duration = time.time() - start_time
        logger.info(f"Agent 13: Successfully completed in {total_duration:.2f}s")
        
        return {
            "success": True,
            "message": "narrative_skeleton_planner completed successfully",
            "data": result.model_dump()
        }

    except Exception as e:
        error_msg = f"Agent 13 failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": error_msg,
            "error": str(e)
        }
