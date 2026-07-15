import sys
import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

import google.genai as genai

# Add paths to orchestrator and audio_design_agent
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, '..'))
from revision_utils import build_revision_prompt_prefix

sys.path.insert(0, os.path.join(base_dir, '..', 'audio_design_agent'))
from audio_design_agent import run_audio_design_agent

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

async def run_reviewer_7(project_id: str, db: Any) -> Dict[str, Any]:
    """
    Reviewer agent for audio design (Agent 7).
    Evaluates audio design assignments against rules and constraints.
    If violations are found, triggers a re-run of audio design agent.
    """
    status_value = 'pass'
    result = None

    try:
        script_doc = await db[SCRIPT_COLLECTION].find_one({'project_id': project_id})
        if not script_doc:
            logger.error(f"Script document not found for project_id: {project_id}")
            return {'reviewer_7_status': 'error'}

        audio_design = script_doc.get('audio_design', {})
        audio_windows = audio_design.get('windows', []) if isinstance(audio_design, dict) else []
        sound_off_compliant = audio_design.get('sound_off_compliant', True) if isinstance(audio_design, dict) else True
        vo_script = script_doc.get('vo_script', [])

        prompt = f"""
        You are an audio design reviewer. Evaluate the audio design below
        against the rules below. Return JSON only — no prose, no markdown.

        AUDIO DESIGN WINDOWS:
        {json.dumps(audio_windows, indent=2)}

        SOUND OFF COMPLIANT FLAG:
        {sound_off_compliant}

        VO SCRIPT (for collision check):
        {json.dumps(vo_script, indent=2)}

        CHECKS — flag a failure for each violation found:
        1. Any music_directive field that contains a specific artist name,
           album name, or track title. Mood and instrumentation directives
           are permitted — named works are not. Flag with window_id.
        2. Any window where the vo_script has silent=false (spoken VO present)
           AND the music_directive for that same window_id contains high-energy
           language such as 'peak', 'driving', 'intense', 'pounding', 'full',
           'climactic'. Music must recede on VO windows. Flag with window_id.
        3. sound_off_compliant is false. This is a global failure.
           Flag with window_id=null and issue='sound_off_compliant is false'.

        If no violations: return {{"pass": true, "failures": []}}
        If violations: return {{"pass": false, "failures": [
          {{"window_id": "W01", "issue": "...", "fix": "..."}}
        ]}}
        """

        # Model Invocation (STRICT SYNTAX REQUIRED)
        invoke_start = time.time()
        logger.info(f"Reviewer 7: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Reviewer 7: Gemini Client instantiated. Sending prompt...")

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
        logger.info(f"Reviewer 7: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Reviewer 7: Raw response length={len(response.text)} chars")
        
        result_data = json.loads(response.text)
        result = ReviewResult.model_validate(result_data)

        if not result.pass_:
            status_value = 'fail_and_re-run'
            
            # Use orchestrator to build revision prompt prefix if available
            revision_prefix = build_revision_prompt_prefix(
                cycle=1,
                target_agent=7,
                instructions=[
                    {'window_id': f.window_id, 'instruction': f.fix, 'source': 'reviewer_7'}
                    for f in result.failures
                ]
            )

            # Trigger re-run of agent 7
            await run_audio_design_agent(project_id, db, revision_prefix=revision_prefix)
        else:
            status_value = 'pass'

    except Exception as e:
        logger.exception(f"Error in run_reviewer_7: {str(e)}")
        status_value = 'error'

    # MongoDB Writes (after review, before return):
    try:
        await db[SCRIPT_COLLECTION].update_one(
            {'project_id': project_id},
            {'$set': {'reviewer_7_status': status_value}}
        )
        
        # Determine failures count to log
        failures_count = 0
        if result and hasattr(result, 'failures'):
            failures_count = len(result.failures)
            
        await db[PIPELINE_COLLECTION].update_one(
            {'project_id': project_id},
            {'$push': {'agent_logs': {
                'agent': 'reviewer_7',
                'status': status_value,
                'failures_count': failures_count,
                'timestamp': time.time()
            }}}
        )
    except Exception as db_e:
        logger.error(f"Failed to write to MongoDB: {str(db_e)}")

    return {'reviewer_7_status': status_value}

if __name__ == "__main__":
    pass