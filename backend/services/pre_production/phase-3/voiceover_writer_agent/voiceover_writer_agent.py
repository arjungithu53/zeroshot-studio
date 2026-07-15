import json
import logging
import os
import time
from typing import List, Optional, Dict, Any

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

logger = logging.getLogger("zeroshot.phase3.voiceover_writer_agent")

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

class VoiceoverWindowLine(BaseModel):
    window_id: str = Field(description="Window identifier, e.g. W01")
    line: Optional[str] = Field(description="exact spoken words, or null for silent window")
    word_count: int = Field(description="count of words in the line")
    silent: bool = Field(description="whether the window is completely silent for voiceover")

class VoiceoverWriterResult(BaseModel):
    vo_script: List[VoiceoverWindowLine]
    reasoning: str
    straight_version_log: str
    lateral_craft_log: str
    status: str = "pending"

async def run_voiceover_writer_agent(project_id: str, db, revision_prefix: str | None = None) -> VoiceoverWriterResult:
    agent_key = "voiceover_writer_agent"
    run_start = time.time()
    logger.info(f"Initializing Agent [5]... | project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error("Agent [5]: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"Agent [5]: Fetching data for project_id={project_id}")
        
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
        concept_hook = concept.get("concept_hook", "")
        cta_framing = concept.get("cta_framing", "")
        archetype = concept.get("archetype", "")
        virality_lever = concept.get("virality_lever", "")

        format_group = phase_2_output.get("format_group", ideation_doc.get("format_group", "N"))
        is_visual = format_group == "V"
        if is_visual:
            selected_motif = phase_2_output.get("selected_visual_motif", ideation_doc.get("selected_visual_motif", {}))
            micro_policy = selected_motif.get("visual_micro_policy", "")
            failure_modes = selected_motif.get("failure_modes", [])
        else:
            selected_archetype = phase_2_output.get("selected_archetype", ideation_doc.get("selected_archetype", {}))
            micro_policy = selected_archetype.get("micro_policy", "")
            failure_modes = selected_archetype.get("failure_modes", [])

        brand_guardrails = ideation_doc.get("brand_guardrails", {})
        tonal_guardrails = brand_guardrails.get("tonal_guardrails", [])
        cultural_modulations = brand_guardrails.get("cultural_modulations", [])
        mandatory_implications = brand_guardrails.get("mandatory_implications", [])

        product_details = project_doc.get("product_details", "")
        target_audience = project_doc.get("target_audience", {})
        brand_guidelines = project_doc.get("brand_guidelines", "")
        product_url = project_doc.get("product_url") or "Not provided"

        video_type_final = ideation_doc.get("video_type_final", project_doc.get("video_type", ""))
        video_type_conditioning_notes = ideation_doc.get("video_type_conditioning_notes", "")

        offer_constraints = script_doc.get("offer_constraints", {})
        vo_prohibition = offer_constraints.get("vo_prohibition", "")
        vo_language_directive = offer_constraints.get("vo_language_directive", "")
        text_super_max_words = offer_constraints.get("text_super_max_words", 0)

        av_channel_map = script_doc.get("av_channel_map", [])
        if not av_channel_map:
            raise ValueError("av_channel_map missing from script document")

        master_timeline_windows = script_doc.get("master_timeline", {}).get("windows", [])
        window_timing = [
            {
                "window_id": w.get("window_id"),
                "start_s": w.get("start_s"),
                "end_s": w.get("end_s"),
                "duration_s": round((w.get("end_s") or 0) - (w.get("start_s") or 0), 2),
                "max_words_at_3_5_wps": int(
                    ((w.get("end_s") or 0) - (w.get("start_s") or 0)) * 3.5
                ),
            }
            for w in master_timeline_windows
        ]

        logger.info(
            f"Agent [5]: Extracted key inputs: concept_id='{concept_id}', "
            f"av_channel_map_count={len(av_channel_map)}, archetype='{archetype}'"
        )
        
        prompt = f"""You are a voiceover copywriter for short-form branded video. You write lines that are both clever and clear — hitting the sweet spot where the strategy is not lost in the cleverness and the cleverness is not lost in the strategy. Your method is: first say it straight, then say it great. You begin with the literal key message for each window and spin it lateral — shorter, more attitudinal, more memorable — without losing the strategic intent.

You write as the brand would talk if it were a person. Not a sales pitch. Not a press release. A voice. You never use exclamation points. You never pre-ramble. The first word of the first line must earn its place. You are grounded strictly in the av_channel_map, offer constraints, and brand directives provided.

You are writing the voiceover script for concept: {concept_id}

CONCEPT HOOK (the opening line register):
{concept_hook}

CTA FRAMING (the register for the offer window):
{cta_framing}

ARCHETYPE:
{archetype}

VIRALITY MECHANIC TO EMBED:
{virality_lever}

ARCHETYPE MICRO-POLICY (governs tone throughout):
{micro_policy}
Failure modes to avoid: {json.dumps(failure_modes)}

BRAND GUARDRAILS:
Tonal guardrails: {json.dumps(tonal_guardrails)}
Cultural modulations: {json.dumps(cultural_modulations)}
Mandatory implications: {json.dumps(mandatory_implications)}

PRODUCT:
{product_details}

PRODUCT WEBPAGE URL:
{product_url}
(If a product URL is provided, use it to research the specific product's features and language in depth before writing.)

TARGET AUDIENCE:
{json.dumps(target_audience)}

BRAND GUIDELINES (if provided):
{brand_guidelines}

FORMAT TYPE: {video_type_final}
FORMAT CONDITIONING DIRECTIVES:
{video_type_conditioning_notes}

Apply these directives to every line you write:
- UGC / Organic-style: The creator is speaking directly to camera — this is NOT a traditional voiceover. Write in first-person present tense as if the creator is talking to their phone. Lines must feel unscripted: use contractions, incomplete thoughts, natural pauses. No brand-voice polish. No sales register. Avoid any phrasing that would sound read off a card. The hook must feel like the creator just started talking mid-thought.
- Testimonial / Real person: Ground every line in a specific felt experience. At least one line must contain a concrete, personal result claim. Emotional sincerity overrides cleverness.
- Animation / Illustrated: Tone can escalate to epic or hyperbolic. Metaphor is permitted and encouraged.
- Narrative / Mini-film: Lines can be slower and more literary. Offer framing is compressed to the final window only.
- Satire / Comedy: Lean into the absurdity of the enemy. Brand lines are the calm, straight-faced punchline.
- Product Beauty: VO is minimal — treat silence as an asset. When copy exists, it is poetic, sensory, and short. Name ingredients and textures as if they are precious. Never explain what the viewer can already see. Maximum 4–5 words per window. The product speaks; the copy whispers.
- Flatlay: VO is primarily informational — ingredients, claims, and key benefits read cleanly over a still composition. Short, declarative lines. No emotional narrative arc; the visual does that work. Copy can be used as text supers instead of spoken VO where appropriate.
- CGI/3D Product: VO is brand-world narration, not product description. Write like a luxury house — evocative, spacious, minimal. Lines should feel like captions in a museum, not a product page. Avoid any line that describes what the visual already shows. The brand name or product name earns one, precise, final mention.

WINDOW TIMING (enforced — do not exceed max_words_at_3_5_wps for any window):
{json.dumps(window_timing, indent=2)}

CHANNEL MAP (what each window's VO channel must communicate):
{json.dumps(av_channel_map, indent=2)}

OFFER WINDOW VO RULES (non-negotiable):
VO prohibition: {vo_prohibition}
VO language directive: {vo_language_directive}
Text super max words: {text_super_max_words}

TASK:
Step 1 — For each window where av_channel_map.vo_info is non-null, write the straight (literal) version of that message in one plain sentence. Log it internally.
Step 2 — Spin it lateral. Shorten it. Add attitude. Say it in the brand's register — mythic, intimate, grounded, never sales-convention. The final line must pass this test: would someone repeat this to a friend?
Step 3 — Count the words. For each window, verify: word_count ≤ max_words_at_3_5_wps from the WINDOW TIMING table above. For windows shorter than 1 second, aim for 0–2 words maximum. A window marked silent must have word_count of 0 and line of null.
Step 4 — For the offer window, apply the VO prohibition and language directive without exception. The virality_lever must activate in this window — the language must structurally embed the sharing mechanic (e.g. "split this with your girl gang" if the mechanic is WhatsApp group share), not just mention it decoratively.
Step 5 — Verify no exclamation points appear anywhere. Verify the first VO line begins mid-thought, not with a setup or greeting.

CONSTRAINTS (apply last):
- Never describe what the visual is showing. The av_channel_map already handles separation.
- Never use clinical, transactional, or pharmacy-register language. Check against tonal_guardrails.
- Silent windows must remain silent — do not add VO to windows marked silent in av_channel_map.
- Output as a JSON array, one entry per window_id, with fields: window_id, line (null if silent), word_count, silent (boolean).
        """

        invoke_start = time.time()
        logger.info(f"Agent [5]: Preparing to call Gemini model={GEMINI_MODEL}...")
        
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent [5]: Gemini Client instantiated. Sending prompt...")
        
        if revision_prefix:
            existing_script_doc = await db[SCRIPT_COLLECTION].find_one(
                {"project_id": str(project_id)}, {"vo_script": 1}
            )
            previous_vo = (existing_script_doc or {}).get("vo_script") or []
            if previous_vo:
                revision_block = (
                    revision_prefix
                    + "\n\nYOUR PREVIOUS VO OUTPUT — copy every window VERBATIM except the windows listed above:\n"
                    + json.dumps(previous_vo, indent=2)
                )
            else:
                revision_block = revision_prefix
            prompt = revision_block + "\n\n" + prompt

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": VoiceoverWriterResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
  
        api_duration = time.time() - invoke_start
        logger.info(f"Agent [5]: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent [5]: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent [5]: Successfully parsed JSON response.")

        result = VoiceoverWriterResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent [5]: Successfully validated structured output with Pydantic.")

        logger.info(f"Agent [5]: Updating SCRIPT and PIPELINE collections...")
        
        # Update SCRIPT_COLLECTION with the final vo_script
        vo_script_dicts = [item.model_dump() for item in result.vo_script]
        await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "vo_script": vo_script_dicts
            }}
        )

        # Update PIPELINE_COLLECTION with logs
        pipeline_log = {
            "agent_name": agent_key,
            "execution_time": run_start,
            "duration": time.time() - run_start,
            "reasoning": result.reasoning,
            "straight_version_log": result.straight_version_log,
            "lateral_craft_log": result.lateral_craft_log,
            "status": result.status
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        total_duration = time.time() - run_start
        logger.info(f"Agent [5]: Completed successfully in {total_duration:.2f}s. Project={project_id}")

        return result

    except Exception as e:
        logger.error(f"Agent [5]: Failed to process project '{project_id}': {e}", exc_info=True)
        raise