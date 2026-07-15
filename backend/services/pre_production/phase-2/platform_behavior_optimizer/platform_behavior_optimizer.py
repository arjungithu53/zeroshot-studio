import json
import logging
import os
import time
from typing import List, Literal

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.platform_behavior_optimizer")

# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------
class PlatformBehaviorOptimizerResult(BaseModel):
    # ideation.platform_rules fields
    platform: str = Field(description="The inferred distribution platform")
    hook_window_rule: str = Field(description="Hook window physics rule based on platform")
    authenticity_signal: Literal["low", "medium", "high"] = Field(description="Authenticity signal level expected by the platform")
    soft_rules: List[str] = Field(description="3-5 soft rules governing this platform's concept physics")
    
    # pipeline log fields
    reasoning: str = Field(description="Reasoning behind inferred platform choices and rules")
    platform_inference_log: str = Field(description="Detailed log tracking inference steps based on media habits and video type")
    platform_variant_seeds: List[str] = Field(description="Variant seeds customized for this distribution platform")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """
This agent translates platform-specific attention dynamics into concept-level physics rules. It operates at the idea level — not the execution or shot level. The output governs how bold or restrained hooks can be, how authentic vs. produced the feel must be, and what virality mechanic is most native to the distribution environment.

Platform rules by distribution context:
Meta Feed (Facebook/Instagram): Hook window = 0–2 seconds. Microhook + immediate product hint required. Face-time minimum: 60% of runtime. Satire tolerance: medium. Authenticity signal: medium.
Instagram Reel: Hook window = 0–1.5 seconds. Trend-aware hooks perform better. Authenticity signal: high — organic feel preferred. Sound-off first design required. Loop-ability is a bonus metric.
YouTube In-stream (skippable): Hook window = 0–5 seconds before skip. Premium production acceptable. Can sustain slower emotional build. Offer must appear before 80% of runtime. Authenticity signal: low.
WhatsApp / Sharing context: Concept must have inherent shareability. Social currency principle applies: the viewer must feel smarter, more informed, or more culturally tuned-in by sharing. Practical value or cultural trigger mechanic required.

Input Context:
- persona.media_habits: {media_habits}
- video_length_seconds: {video_length_seconds}
- video_type_final: {video_type_final}
- priority_directives: {priority_directives}

Based on the provided persona.media_habits and video_type_final, infer the most likely distribution platform. Produce the platform_rules object with: platform name, hook_window_rule, authenticity_signal (low/medium/high), and 3–5 soft rules governing this platform's concept physics.
Also produce the pipeline logs detailing your reasoning, an inference log, and 3-5 platform variant seeds.
"""

def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------
async def run_platform_behavior_optimizer_agent(project_id: str, db) -> PlatformBehaviorOptimizerResult:
    agent_key = "platform_behavior_optimizer"
    logger.info(f"Initializing Agent [{agent_key}]... project_id={project_id}")
    
    start_time = time.time()
    
    try:
        # Fetch Data
        logger.info(f"Fetching data for project_id={project_id}")
        
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")
            
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        if not strategy_doc:
            raise ValueError(f"Strategy document for '{project_id}' not found")
            
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")
            
        # Extract fields cautiously
        video_length_seconds = project_doc.get("video_length_seconds", 30)
        
        # Audience Persona media habits
        audience_persona = strategy_doc.get("strategy", {}).get("agents", {}).get("audience_persona", {})
        media_habits = audience_persona.get("media_habits", [])
        
        # Ideation outputs
        video_type_final = ideation_doc.get("video_type_final", "Unknown")
        priority_directives = ideation_doc.get("priority_directives", {})
        
        logger.info(f"Agent [{agent_key}]: Extracted inputs -> media_habits={media_habits}, video_length_seconds={video_length_seconds}, video_type_final={video_type_final}")

        prompt = PROMPT_TEMPLATE.format(
            media_habits=json.dumps(media_habits, indent=2),
            video_length_seconds=video_length_seconds,
            video_type_final=video_type_final,
            priority_directives=json.dumps(priority_directives, indent=2)
        )
        
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
                "response_json_schema": PlatformBehaviorOptimizerResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent [{agent_key}]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent [{agent_key}]: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent [{agent_key}]: Successfully parsed JSON response.")

        result = PlatformBehaviorOptimizerResult(**parsed_data)
        logger.info(f"Agent [{agent_key}]: Successfully validated structured output with Pydantic.")

        # Update DB
        logger.info(f"Updating IDEATION and PIPELINE collections for project_id={project_id}")
        platform_rules = {
            "platform": getattr(result, 'platform', ''),
            "hook_window_rule": getattr(result, 'hook_window_rule', ''),
            "authenticity_signal": getattr(result, 'authenticity_signal', 'medium'),
            "soft_rules": getattr(result, 'soft_rules', [])
        }
        
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {"platform_rules": platform_rules}}
        )
        
        pipeline_log = {
            "agent_id": 15,
            "agent_name": agent_key,
            "duration": time.time() - start_time,
            "api_duration": api_duration,
            "reasoning": getattr(result, 'reasoning', ''),
            "platform_inference_log": getattr(result, 'platform_inference_log', ''),
            "platform_variant_seeds": getattr(result, 'platform_variant_seeds', []),
            "timestamp": time.time()
        }
        
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        total_duration = time.time() - start_time
        logger.info(f"Agent [{agent_key}] execution completed in {total_duration:.2f}s")
        
        return result

    except Exception as e:
        logger.error(f"Agent [{agent_key}] execution failed: {str(e)}")
        raise
