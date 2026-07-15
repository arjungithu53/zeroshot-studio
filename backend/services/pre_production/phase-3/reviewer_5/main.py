import sys
import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

import google.genai as genai

# Add paths to orchestrator and voiceover_writer_agent
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, '..'))
from revision_utils import build_revision_prompt_prefix

sys.path.insert(0, os.path.join(base_dir, '..', 'voiceover_writer_agent'))
from voiceover_writer_agent import run_voiceover_writer_agent

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

async def run_reviewer_5(project_id: str, db: Any) -> Dict[str, Any]:
    """
    Reviewer agent for voiceover writer (Agent 5).
    Evaluates the VO script against constraints and tonal guardrails.
    If violations are found, triggers a re-run of voiceover writer agent.
    """
    status_value = 'pass'
    result = None

    try:
        script_doc = await db[SCRIPT_COLLECTION].find_one({'project_id': project_id})
        ideation_doc = await db[IDEATION_COLLECTION].find_one({'project_id': project_id})

        if not script_doc or not ideation_doc:
            logger.error(f"Required documents not found for project_id: {project_id}")
            return {'reviewer_5_status': 'error'}

        vo_script = script_doc.get('vo_script', [])
        phase_2_output = ideation_doc.get('phase_2_output', {})
        format_group = phase_2_output.get('format_group', ideation_doc.get('format_group', 'N'))
        if format_group == 'V':
            selected_motif = phase_2_output.get('selected_visual_motif', ideation_doc.get('selected_visual_motif', {}))
            micro_policy = selected_motif.get('visual_micro_policy', '')
            failure_modes = selected_motif.get('failure_modes', [])
        else:
            selected_arch = phase_2_output.get('selected_archetype', {})
            micro_policy = selected_arch.get('micro_policy', '')
            failure_modes = selected_arch.get('failure_modes', [])
        
        brand_guardrails = ideation_doc.get('brand_guardrails', {})
        tonal_guardrails = brand_guardrails.get('tonal_guardrails', [])

        prompt = f"""
        You are a voiceover script reviewer. Evaluate the VO script below
        against the rules below. Return JSON only — no prose, no markdown.

        VO SCRIPT:
        {json.dumps(vo_script, indent=2)}

        ARCHETYPE MICRO-POLICY:
        {micro_policy}

        ARCHETYPE FAILURE MODES TO AVOID:
        {json.dumps(failure_modes, indent=2)}

        TONAL GUARDRAILS:
        {json.dumps(tonal_guardrails, indent=2)}

        CHECKS — flag a failure for each violation found:
        1. Any line containing clinical, pharmaceutical, or transactional
           register language e.g. 'moisturizes', 'dermatologically tested',
           'buy now', 'limited time', 'clinically proven', 'recommended by'.
           Flag with window_id and the offending phrase.
        2. Any line that matches a pattern described in failure_modes.
           Flag with window_id and which failure mode it matches.
        3. Any line containing an exclamation point. Flag with window_id.
        4. Any window where silent=true but line is not null.
           Flag with window_id — this is a channel conflict.
        5. The first non-silent VO line (lowest window_id with silent=false)
           starting with a setup or greeting rather than mid-thought.
           Setup words: 'Hey', 'Hi', 'So,', 'Welcome', 'Introducing',
           'Today', 'Let me', 'Have you'. Flag with window_id.

        If no violations: return {{"pass": true, "failures": []}}
        If violations: return {{"pass": false, "failures": [
          {{"window_id": "W01", "issue": "...", "fix": "..."}}
        ]}}
        """
        
        # Model Invocation (STRICT SYNTAX REQUIRED)
        invoke_start = time.time()
        logger.info(f"Reviewer 5: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Reviewer 5: Gemini Client instantiated. Sending prompt...")

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
        logger.info(f"Reviewer 5: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Reviewer 5: Raw response length={len(response.text)} chars")
        
        result_data = json.loads(response.text)
        result = ReviewResult.model_validate(result_data)

        if not result.pass_:
            status_value = 'fail_and_re-run'
            
            # Use orchestrator to build revision prompt prefix if available
            revision_prefix = build_revision_prompt_prefix(
                cycle=1,
                target_agent=5,
                instructions=[
                    {'window_id': f.window_id, 'instruction': f.fix, 'source': 'reviewer_5'}
                    for f in result.failures
                ]
            )

            # Trigger re-run of agent 5
            await run_voiceover_writer_agent(project_id, db, revision_prefix=revision_prefix)
        else:
            status_value = 'pass'

    except Exception as e:
        logger.exception(f"Error in run_reviewer_5: {str(e)}")
        status_value = 'error'

    # MongoDB Writes (after review, before return):
    try:
        await db[SCRIPT_COLLECTION].update_one(
            {'project_id': project_id},
            {'$set': {'reviewer_5_status': status_value}}
        )
        
        # Determine failures count to log
        failures_count = 0
        if result and hasattr(result, 'failures'):
            failures_count = len(result.failures)
            
        await db[PIPELINE_COLLECTION].update_one(
            {'project_id': project_id},
            {'$push': {'agent_logs': {
                'agent': 'reviewer_5',
                'status': status_value,
                'failures_count': failures_count,
                'timestamp': time.time()
            }}}
        )
    except Exception as db_e:
        logger.error(f"Failed to write to MongoDB: {str(db_e)}")

    return {'reviewer_5_status': status_value}

if __name__ == "__main__":
    pass
