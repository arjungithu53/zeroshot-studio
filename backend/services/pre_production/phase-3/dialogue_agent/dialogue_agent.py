import json
import logging
import os
import time
from typing import List, Optional, Dict, Any

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

logger = logging.getLogger("zeroshot.phase3.dialogue_agent")

def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")

class DialogueLine(BaseModel):
    window_id: str = Field(description="Window identifier, e.g. W01")
    character: str = Field(description="The character speaking the line")
    line: str = Field(description="The spoken dialogue line")

class DialogueResult(BaseModel):
    dialogue_lines: Optional[List[DialogueLine]] = Field(description="List of dialogue lines, or null if no dialogue is required")
    reasoning: str = Field(description="Explanation for the dialogue choices")
    naturalness_review: str = Field(description="Review of why the speech rhythm feels authentic and non-salesy")
    fragment_analysis: str = Field(description="Analysis of sentence fragments and interruptions used")
    status: str = "pending"

async def run_dialogue_agent(project_id: str, db, revision_prefix: str | None = None) -> DialogueResult:
    agent_key = "dialogue_agent"
    run_start = time.time()
    logger.info(f"Initializing Agent [6]... | project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error("Agent [6]: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"Agent [6]: Fetching data for project_id={project_id}")
        
        # 1. Fetch Projects
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")
            
        # 2. Fetch Ideation
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")
            
        # 3. Fetch Script
        script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
        if not script_doc:
            raise ValueError(f"Script document for '{project_id}' not found")

        # Extract Inputs
        phase_2_output = ideation_doc.get("phase_2_output", ideation_doc)
        approved_concepts = phase_2_output.get("approved_concepts", ideation_doc.get("approved_concepts", []))
        if not approved_concepts:
            raise ValueError("No approved concepts found in ideation document")

        concept = approved_concepts[0]
        concept_id = concept.get("concept_id", "Unknown Concept")
        product_url = project_doc.get("product_url") or "Not provided"

        format_group = phase_2_output.get("format_group", ideation_doc.get("format_group", "N"))
        is_visual = format_group == "V"
        beats = concept.get("composition_beats", []) if is_visual else concept.get("story_beats", [])
        archetype = concept.get("visual_motif", "") if is_visual else concept.get("archetype", "")

        if is_visual:
            selected_motif = phase_2_output.get("selected_visual_motif", ideation_doc.get("selected_visual_motif", {}))
            micro_policy = selected_motif.get("visual_micro_policy", "")
        else:
            selected_archetype = phase_2_output.get("selected_archetype", ideation_doc.get("selected_archetype", {}))
            micro_policy = selected_archetype.get("micro_policy", "")

        video_type_final = ideation_doc.get("video_type_final", project_doc.get("video_type", ""))
        video_type_conditioning_notes = ideation_doc.get("video_type_conditioning_notes", "")

        vo_script = script_doc.get("vo_script", [])

        logger.info(
            f"Agent [6]: Extracted key inputs: concept_id='{concept_id}', format_group={format_group}, "
            f"beats_count={len(beats)}, archetype/motif='{archetype}'"
        )

        prompt = f"""
You are a dialogue writer for short-form video. You write words meant to be spoken by real people — which means they must sound like real people speaking, not like scripts. Real people use sentence fragments. Real people step on each other's lines. Real people start a thought, restart it, trail off. Your job is to capture that authenticity while ensuring the sales message never appears in a character's mouth. If a concept beat requires a product claim, that belongs in VO, not in dialogue.

If no dialogue is required by the concept, you output null immediately and do not invent dialogue to fill space. You are grounded strictly in the beats and VO script provided.
You are evaluating dialogue requirements for concept: {concept_id}

PRODUCT WEBPAGE URL:
{product_url}
(If a product URL is provided, use it to understand what the product is so that you correctly exclude product claims from dialogue and route them to VO.)

{"COMPOSITION BEATS" if is_visual else "STORY BEATS"}:
{json.dumps(beats, indent=2)}

{"VISUAL MOTIF" if is_visual else "ARCHETYPE"}:
{archetype}

{"VISUAL MOTIF MICRO-POLICY" if is_visual else "ARCHETYPE MICRO-POLICY"}:
{micro_policy}

FORMAT TYPE: {video_type_final}
FORMAT CONDITIONING DIRECTIVES:
{video_type_conditioning_notes}

Apply these directives when deciding whether dialogue exists and how to write it:
- UGC / Organic-style: The creator speaks directly to camera — this is NOT a multi-person scene. In almost all UGC concepts, dialogue should be null. Only write dialogue if a beat explicitly requires a second person responding on screen (e.g. a friend reacting). Never invent a second character to fill space.
- Testimonial / Real person: Same as UGC — single person to camera. Dialogue null unless a second character is structurally required.
- Narrative / Mini-film: Multi-character dialogue is permitted and should feel naturalistic.
- Satire / Comedy: Dialogue can include an exaggerated antagonist voice. Keep it short and punchy — the enemy's lines exist to be undercut.
- Product Beauty: No human talent, therefore dialogue is always null. Output null immediately.
- Flatlay: No human talent, therefore dialogue is always null. Output null immediately.
- CGI/3D Product: No human talent, therefore dialogue is always null. Output null immediately.

VO SCRIPT (already written — dialogue must not duplicate this):
{json.dumps(vo_script, indent=2)}

TASK:
Step 1 — Read through all story beats. Determine whether any beat explicitly requires character-to-character speech (i.e. two or more people speaking to each other, not a creator speaking to camera). If no beat requires this, output null and stop.

Step 2 — If dialogue is required: identify which window(s) it appears in, who is speaking, and what the dramatic function of the exchange is.

Step 3 — Write the dialogue. Apply these rules without exception:
- Use sentence fragments and natural speech patterns.
- Allow one character to complete the other's sentence or interrupt.
- Never embed a product claim, price point, or offer language in dialogue — that belongs in VO or text super only.
- Where the talent would benefit from ad-libbing, write a direction note (e.g. "[ad-lib reaction]") rather than over-scripting every word.

Step 4 — Read each line aloud mentally. Ask: does this sound like a person or a press release? Revise any line that sounds like a press release.

CONSTRAINTS (apply last):
- If no dialogue is required, the output must be null — do not invent dialogue.
- Dialogue must not repeat or duplicate anything already in the vo_script for the same window.
- Output as a JSON array of objects with fields: window_id, character, line — or null if no dialogue is required."""

        invoke_start = time.time()
        logger.info(f"Agent [6]: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent [6]: Gemini Client instantiated. Sending prompt...")
        
        if revision_prefix:
            existing_script_doc = await db[SCRIPT_COLLECTION].find_one(
                {"project_id": str(project_id)}, {"dialogue_lines": 1}
            )
            previous_lines = (existing_script_doc or {}).get("dialogue_lines") or []
            if previous_lines:
                revision_block = (
                    revision_prefix
                    + "\n\nYOUR PREVIOUS DIALOGUE OUTPUT — copy every line VERBATIM except the windows listed above:\n"
                    + json.dumps(previous_lines, indent=2)
                )
            else:
                revision_block = revision_prefix
            prompt = revision_block + "\n\n" + prompt

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": DialogueResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
  
        api_duration = time.time() - invoke_start
        logger.info(f"Agent [6]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent [6]: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent [6]: Successfully parsed JSON response.")

        result = DialogueResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent [6]: Successfully validated structured output with Pydantic.")

        # Update SCRIPT and PIPELINE collections
        logger.info(f"Agent [6]: Updating collections for project_id={project_id}")
        
        lines_data = [l.model_dump() for l in result.dialogue_lines] if result.dialogue_lines else None
        
        await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {"dialogue_lines": lines_data}}
        )
        
        pipeline_log = {
            "agent_name": agent_key,
            "execution_time_sec": round(time.time() - run_start, 2),
            "reasoning": result.reasoning,
            "naturalness_review": result.naturalness_review,
            "fragment_analysis": result.fragment_analysis,
            "timestamp": time.time()
        }
        
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        total_duration = time.time() - run_start
        logger.info(f"Agent [6]: Completed successfully in {total_duration:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Agent [6]: Error during execution: {str(e)}", exc_info=True)
        raise