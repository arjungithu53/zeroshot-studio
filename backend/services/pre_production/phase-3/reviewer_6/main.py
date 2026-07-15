import sys
import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

import google.genai as genai

# Add paths to orchestrator and dialogue_agent
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, '..'))
from revision_utils import build_revision_prompt_prefix

sys.path.insert(0, os.path.join(base_dir, '..', 'dialogue_agent'))
from dialogue_agent import run_dialogue_agent

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")

class ReviewFailure(BaseModel):
    window_id:  str | None = None
    issue:      str
    fix:        str

class ReviewResult(BaseModel):
    pass_:      bool = Field(alias='pass')
    failures:   list[ReviewFailure] = Field(default_factory=list)

async def run_reviewer_6(project_id: str, db: Any) -> Dict[str, Any]:
    """
    Reviewer agent for dialogue writer (Agent 6).
    Evaluates dialogue lines against constraints and rules.
    If violations are found, triggers a re-run of dialogue agent.
    """
    status_value = 'pass'
    result = None

    try:
        script_doc = await db[SCRIPT_COLLECTION].find_one({'project_id': project_id})
        if not script_doc:
            logger.error(f"Script document not found for project_id: {project_id}")
            return {'reviewer_6_status': 'error'}

        dialogue_lines = script_doc.get('dialogue_lines')
        if dialogue_lines is None:
            # Agent 6 correctly determined no dialogue is required.
            # Do NOT call Gemini. Return pass immediately.
            logger.info(f"Reviewer 6: No dialogue lines found for project_id: {project_id}, passing via null logic.")
            await db[SCRIPT_COLLECTION].update_one(
                {'project_id': project_id},
                {'$set': {'reviewer_6_status': 'pass_null_dialogue'}}
            )
            return {'reviewer_6_status': 'pass_null_dialogue'}

        vo_script = script_doc.get('vo_script', [])

        prompt = f"""
        You are a dialogue reviewer. Evaluate the dialogue lines below
        against the rules below. Return JSON only — no prose, no markdown.

        DIALOGUE LINES:
        {json.dumps(dialogue_lines, indent=2)}

        VO SCRIPT (dialogue must not duplicate VO in the same window):
        {json.dumps(vo_script, indent=2)}

        CHECKS — flag a failure for each violation found:
        1. Any dialogue line that contains a product name, price point,
           offer detail, or brand benefit claim. These belong in VO only.
           Flag with window_id and the offending content.
        2. Any dialogue line whose content duplicates or closely paraphrases
           the vo_script line for the same window_id.
           Flag with window_id.
        3. Any dialogue line that reads like a press release rather than
           natural speech — complete grammatical sentences with no fragments,
           no interruptions, no trailing off, no natural hesitations.
           Flag with window_id and explain why it sounds scripted.

        If no violations: return {{"pass": true, "failures": []}}
        If violations: return {{"pass": false, "failures": [
          {{"window_id": "W01", "issue": "...", "fix": "..."}}
        ]}}
        """

        # Model Invocation (STRICT SYNTAX REQUIRED)
        invoke_start = time.time()
        logger.info(f"Reviewer 6: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Reviewer 6: Gemini Client instantiated. Sending prompt...")

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
        logger.info(f"Reviewer 6: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Reviewer 6: Raw response length={len(response.text)} chars")
        
        result_data = json.loads(response.text)
        result = ReviewResult.model_validate(result_data)

        if not result.pass_:
            status_value = 'fail_and_re-run'
            
            # Use orchestrator to build revision prompt prefix if available
            revision_prefix = build_revision_prompt_prefix(
                cycle=1,
                target_agent=6,
                instructions=[
                    {'window_id': f.window_id, 'instruction': f.fix, 'source': 'reviewer_6'}
                    for f in result.failures
                ]
            )

            # Trigger re-run of agent 6
            await run_dialogue_agent(project_id, db, revision_prefix=revision_prefix)
        else:
            status_value = 'pass'

    except Exception as e:
        logger.exception(f"Error in run_reviewer_6: {str(e)}")
        status_value = 'error'

    # MongoDB Writes (after review, before return):
    try:
        await db[SCRIPT_COLLECTION].update_one(
            {'project_id': project_id},
            {'$set': {'reviewer_6_status': status_value}}
        )
        
        # Determine failures count to log
        failures_count = 0
        if result and hasattr(result, 'failures'):
            failures_count = len(result.failures)
            
        await db[PIPELINE_COLLECTION].update_one(
            {'project_id': project_id},
            {'$push': {'agent_logs': {
                'agent': 'reviewer_6',
                'status': status_value,
                'failures_count': failures_count,
                'timestamp': time.time()
            }}}
        )
    except Exception as db_e:
        logger.error(f"Failed to write to MongoDB: {str(db_e)}")

    return {'reviewer_6_status': status_value}

if __name__ == "__main__":
    pass