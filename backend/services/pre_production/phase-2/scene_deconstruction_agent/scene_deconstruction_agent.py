import json
import logging
import os
import time
from typing import List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.scene_deconstruction_agent")

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")

# ── Pydantic Models for Structured Output ────────────────────────────────────

class AtomicElements(BaseModel):
    character_action: str = Field(description="What is the person in the scene physically doing?")
    emotional_tone: str = Field(description="What is the character feeling or projecting in this moment?")
    product_interaction_type: str = Field(description="How is the product being used (application, reveal, unboxing, transformation)?")
    environmental_context: str = Field(description="Where is this happening and what does the environment communicate?")
    symbolic_meaning: str = Field(description="What does this scene represent beyond its literal action?")

class SceneDeconstructionResult(BaseModel):
    # Pipeline log fields
    reasoning: str = Field(description="Explanation of the extraction and deconstruction logic")
    non_negotiable_analysis: List[str] = Field(description="Which atomic elements are fixed requirements (the brand explicitly stated them)")
    interpretive_freedoms: List[str] = Field(description="Creative latitude for downstream agents (outcomes described by brand, not specific execution)")
    
    # ideation.scene_intelligence partial write
    status: str = Field(default="completed", description="Execution status of the agent")
    reason: Optional[str] = Field(default=None, description="Reason if skipped")
    scene_text: str = Field(description="The preferred scene text from the project")
    atomic_elements: AtomicElements = Field(description="The broken down atomic narrative elements")

def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

async def run_scene_deconstruction_agent(project_id: str, db) -> SceneDeconstructionResult:
    """
    Agent 7: Scene Deconstruction Agent
    Breaks the preferred scene into five atomic elements: character action, emotional tone,
    product interaction type, environmental context, symbolic meaning. Distinguishes non-negotiable
    elements (brand-fixed) from interpretive freedoms (creative latitude for downstream agents).
    """
    agent_key = "scene_deconstruction_agent"
    logger.info(f"[{agent_key}] Starting agent execution | project_id={project_id}")
    start_time = time.time()

    try:
        # Fetch project data (using ObjectId for the projects collection)
        logger.info(f"[{agent_key}] Fetching projects document for project_id={project_id}")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            logger.error(f"[{agent_key}] Project not found | project_id={project_id}")
            raise ValueError(f"Project '{project_id}' not found")

        preferred_scene = project_doc.get("preferred_scene")
        
        # SKIP CONDITION: IF project.preferred_scene IS NOT NULL | SKIP agents 7-11 if null
        if not preferred_scene:
            logger.info(f"[{agent_key}] SKIP CONDITION MET: project.preferred_scene is null.")
            skipped_result = SceneDeconstructionResult(
                status="skipped",
                reason="project.preferred_scene is null",
                scene_text="",
                atomic_elements=AtomicElements(
                    character_action="",
                    emotional_tone="",
                    product_interaction_type="",
                    environmental_context="",
                    symbolic_meaning=""
                ),
                reasoning="",
                non_negotiable_analysis=[],
                interpretive_freedoms=[]
            )
            
            # Write to DB for skipped run
            await db[IDEATION_COLLECTION].update_one(
                {"project_id": str(project_id)},
                {"$set": {
                    "scene_intelligence": {
                        "status": "skipped",
                        "reason": "project.preferred_scene is null"
                    }
                }},
                upsert=True
            )
            return skipped_result

        # Fetch strategy data (using string project_id for the strategy collection)
        logger.info(f"[{agent_key}] Fetching strategy document for project_id={project_id}")
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        if not strategy_doc:
            logger.error(f"[{agent_key}] Strategy document not found | project_id={project_id}")
            raise ValueError(f"Strategy for project '{project_id}' not found")
            
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        video_type_final = ideation_doc.get("video_type_final", "N/A") if ideation_doc else "N/A"

        campaign_platform = strategy_doc.get("truth_conflict_platform", {}).get("campaign_platform", "N/A")

        prompt = f"""
        You are the Scene Deconstruction Agent. Your purpose is to break the preferred scene into five atomic elements: 
        character action, emotional tone, product interaction type, environmental context, symbolic meaning. 
        You must also distinguish non-negotiable elements (brand-fixed) from interpretive freedoms (creative latitude for downstream agents).

        ## Input Data
        **preferred_scene**: "{preferred_scene}"
        **campaign_platform**: "{campaign_platform}"
        **video_type_final**: "{video_type_final}"

        ## Task Instructions
        1. Break the preferred scene into these five atomic elements:
           - Character action: What is the person in the scene physically doing?
           - Emotional tone: What is the character feeling or projecting in this moment?
           - Product interaction type: How is the product being used (application, reveal, unboxing, transformation)?
           - Environmental context: Where is this happening and what does the environment communicate?
           - Symbolic meaning: What does this scene represent beyond its literal action?
        2. Identify which elements are non-negotiable (explicitly stated by the brand).
        3. Identify interpretive freedoms (outcomes described by brand, but specific execution left to interpretation).
        4. Explain your reasoning.
        """

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
                "response_json_schema": SceneDeconstructionResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent [{agent_key}]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent [{agent_key}]: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent [{agent_key}]: Successfully parsed JSON response.")

        result = SceneDeconstructionResult(**parsed_data)
        result.status = "completed"
        result.scene_text = preferred_scene
        logger.info(f"Agent [{agent_key}]: Successfully validated structured output with Pydantic.")

        logger.info(f"Agent [{agent_key}]: Updating IDEATION and PIPELINE collections...")
        
        # Update Ideation
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "scene_intelligence": {
                    "status": "completed",
                    "reason": None,
                    "scene_text": result.scene_text,
                    "atomic_elements": result.atomic_elements.model_dump()
                }
            }},
            upsert=True
        )

        # Update Pipeline Log
        pipeline_log = {
            "agent_id": "agent_7",
            "agent_name": "scene_deconstruction_agent",
            "timestamp": time.time(),
            "execution_duration": time.time() - start_time,
            "status": "completed",
            "reasoning": result.reasoning,
            "non_negotiable_analysis": result.non_negotiable_analysis,
            "interpretive_freedoms": result.interpretive_freedoms
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        total_duration = time.time() - start_time
        logger.info(f"[{agent_key}] Execution completed successfully in {total_duration:.2f}s | project_id={project_id}")
        return result

    except Exception as e:
        logger.error(f"[{agent_key}] Fatal error during execution | project_id={project_id} | err={e}", exc_info=True)
        raise
