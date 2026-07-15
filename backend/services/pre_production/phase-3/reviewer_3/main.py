import sys
import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

import google.genai as genai

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from revision_utils import build_revision_prompt_prefix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'visual_sequencing_agent'))
from visual_sequencing_agent import run_visual_sequencing_agent

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")

class ReviewFailure(BaseModel):
    window_id:  str | None = None
    issue:      str
    fix:        str

class ReviewResult(BaseModel):
    pass_:      bool = Field(alias='pass')
    failures:   list[ReviewFailure] = Field(default_factory=list)

async def run_reviewer_3(project_id: str, db: Any) -> Dict[str, Any]:
    """
    Reviewer agent for visual sequencing (Agent 3).
    Evaluates the shot list against offer constraints and visual rules.
    If violations are found, triggers a re-run of visual sequencing agent.
    """
    status_value = 'pass'
    result = None

    try:
        script_doc = await db[SCRIPT_COLLECTION].find_one({'project_id': project_id})
        if not script_doc:
            logger.error(f"Script document not found for project_id: {project_id}")
            return {'reviewer_3_status': 'error'}

        shot_list = script_doc.get('shot_list', [])
        offer_constraints = script_doc.get('offer_constraints', {})
        text_super_max_words = offer_constraints.get('text_super_max_words', 8)
        visual_directive = offer_constraints.get('visual_directive', '')

        prompt = f"""
        You are a shot list reviewer. Evaluate the shot list below against
        the rules below. Return JSON only — no prose, no markdown.

        SHOT LIST:
        {json.dumps(shot_list, indent=2)}

        OFFER CONSTRAINTS:
        text_super_max_words: {text_super_max_words}
        visual_directive: {visual_directive}

        CHECKS — flag a failure for each violation found:
        1. Any text_super with more words than text_super_max_words.
           Flag with window_id and exact word count found.
        2. Any shot where the action field contains vague non-actionable language
           e.g. 'something beautiful', 'mythic feel', 'looks nice', 'elegant'.
           Flag with window_id and the offending phrase.
        3. The final window (offer window) must have at least one text_super.
           If it has none, flag it.
        4. Any framing field that describes a horizontal/landscape composition
           incompatible with 9:16 vertical format. Flag with window_id.
        5. AI GENERATOR FEASIBILITY: Descriptions MUST NOT contain impossible physics, high-speed environmental morphing (e.g., 'background suddenly streaks', 'lights fracture into a tunnel'), highly chaotic motion blur, extreme wind/explosions, or impossible camera stunts. Generative AI models fail on these. Flag with window_id and the offending phrase if the action is too computationally complex or unstable for an AI video generator.

        If no violations found: return {{"pass": true, "failures": []}}
        If violations found: return {{"pass": false, "failures": [
          {{"window_id": "W01", "issue": "...", "fix": "..."}}
        ]}}
        """
        
        # Model Invocation (STRICT SYNTAX REQUIRED)
        invoke_start = time.time()
        logger.info(f"Reviewer 3: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Reviewer 3: Gemini Client instantiated. Sending prompt...")

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ReviewResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Reviewer 3: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Reviewer 3: Raw response length={len(response.text)} chars")
        
        result_data = json.loads(response.text)
        result = ReviewResult.model_validate(result_data)

        if not result.pass_:
            status_value = 'fail_and_re-run'
            reviewer_failures = result.failures
            
            # Use orchestrator to build revision prompt prefix if available
            revision_prefix = build_revision_prompt_prefix(
                cycle=1,
                target_agent=3,
                instructions=[
                    {'window_id': f.window_id, 'instruction': f.fix, 'source': 'reviewer_3'}
                    for f in result.failures
                ]
            )

            # Trigger re-run of agent 3
            await run_visual_sequencing_agent(project_id, db, revision_prefix=revision_prefix)
        else:
            status_value = 'pass'

    except Exception as e:
        logger.exception(f"Error in run_reviewer_3: {str(e)}")
        status_value = 'error'

    # MongoDB Writes (after review, before return):
    try:
        await db[SCRIPT_COLLECTION].update_one(
            {'project_id': project_id},
            {'$set': {'reviewer_3_status': status_value}}
        )
        await db[PIPELINE_COLLECTION].update_one(
            {'project_id': project_id},
            {'$push': {'agent_logs': {
                'agent': 'reviewer_3',
                'status': status_value,
                'failures_count': len(result.failures) if result else 0,
                'timestamp': time.time()
            }}}
        )
    except Exception as db_e:
        logger.error(f"Failed to write to MongoDB: {str(db_e)}")

    return {'reviewer_3_status': status_value}

if __name__ == "__main__":
    pass
