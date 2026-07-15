import json
import logging
import os
import time
from typing import List, Optional, Literal

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.brand_guideline_alignment")

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

class MandatoryTranslation(BaseModel):
    mandatory: str
    creative_implication: str

class BrandGuardrails(BaseModel):
    tonal_guardrails: List[str]
    cultural_modulations: List[str]
    mandatory_implications: List[str]

class BrandGuidelineAlignmentResult(BaseModel):
    status: Literal["completed", "skipped", "error"]
    reason: Optional[str] = None
    brand_guardrails: Optional[BrandGuardrails] = None
    reasoning: Optional[str] = None
    mandatory_translation_log: Optional[List[MandatoryTranslation]] = None
    cultural_sensitivity_notes: Optional[str] = None

# ---------------------------------------------------------------------------
# Helper Methods
# ---------------------------------------------------------------------------

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
# Core Agent Logic
# ---------------------------------------------------------------------------

async def run_brand_guideline_alignment_agent(project_id: str, db) -> dict:
    """
    Agent 4: Brand Guideline Alignment
    Translates brand mandatories and adjectives into concrete tonal guardrails 
    and cultural modulation rules. Runs EVERY run regardless of whether idea branch ran.
    """
    agent_key = "brand_guideline_alignment"
    logger.info(f"[{agent_key}] Starting agent execution | project_id={project_id}")

    start_time = time.time()
    try:
        # 1. Fetch data from Projects Collection
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")
            
        # 2. Fetch data from Strategy Collection
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": project_id})
        if not strategy_doc:
            raise ValueError(f"Strategy for project '{project_id}' not found")
            
        # Input Data Extraction
        brand_guidelines = project_doc.get("brand_guidelines")
        product_details = project_doc.get("product_details", "")
        target_audience = project_doc.get("target_audience", {})
        location_context = project_doc.get("location_context")
        category_dominant_tone = strategy_doc.get("competitive_landscape", {}).get("category_dominant_tone", "")
        
        brand_adjective = strategy_doc.get("brand_adjective", "")
        creative_brief = strategy_doc.get("strategy_models", {}).get("creative_brief", {})
        tone_of_voice = creative_brief.get("tone_of_voice", "")
        mandatories = creative_brief.get("mandatories", [])
        campaign_platform = strategy_doc.get("truth_conflict_platform", {}).get("campaign_platform", "")
        positioning_statement = strategy_doc.get("positioning_alignment", {}).get("positioning_statement", "")

        # 3. Construct Prompt
        prompt = f"""You are the Brand Guideline Alignment Agent (Soft Constraint Guardian).
Your purpose is to translate brand mandatories and brand adjectives into concrete tonal guardrails and cultural modulation rules. These will be injected into concept generation agents to keep concepts consistent with the brand identity. This is not a hard compliance filter that eliminates creative directions, but a soft guardian defining rules.

Input Data:
- Brand Adjective: {brand_adjective}
- Tone of Voice: {tone_of_voice}
- Mandatories: {json.dumps(mandatories, indent=2)}
- Campaign Platform: {campaign_platform}
- Positioning Statement: {positioning_statement}
- Brand Guidelines: {brand_guidelines}
- Product Details: {product_details}
- Target Audience: {json.dumps(target_audience, indent=2)}
- Category Dominant Tone (Negative constraint): {category_dominant_tone}
- Location Context: {location_context}

Follow these instructions exactly:

Step 1: Translate Mandatories
Parse the provided mandatories list. Translate each mandatory into a creative implication.
For example, if a mandatory states "no clinical jargon", the implication is "replace efficacy language with sensory or emotional language in all concept copy".
If the mandatories list is empty, derive sensible rules from the Brand Adjective and Tone of Voice.

Step 2: Define Tonal Guardrails
Define three tonal guardrails STRICTLY grounded in the provided Brand Adjective ("{brand_adjective}").
These guardrails must address tone, language, and conceptual boundaries. Ensure you do NOT rely on generic traits; you must anchor these guardrails entirely around how the adjective "{brand_adjective}" should and shouldn't manifest in creative work. Your output must be organically generated from the current project's adjective.

Step 3: Cultural Sensitivities
Identify cultural sensitivities specific to the Target Audience.
Ensure you consider nuanced cultural context without relying on stereotypes.

Step 3b: Negative Tonal Constraint
Use the Category Dominant Tone to define what this brand must actively NOT sound like — construct a negative guardrail as precise as the positive ones.
If the dominant tone is "functional," the negative constraint dictates that concept copy sounding functional fails this guardrail.

Step 3c: Location Context Modulation
Apply the Location Context to modulate climate/location-specific guardrails.
Ensure no imagery or language creates cognitive dissonance with the persona's daily reality in that location context.

Step 4: Final Output
Consolidate these into the exact structured output matching the provided schema.
Set status to "completed" and return STRICTLY in the requested JSON format matching the schema.
"""

        # 4. Generate with Gemini Structuring
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment.")
            
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        
        logger.info(f"[{agent_key}] Sending request to Gemini...")
        api_start_time = time.time()
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": BrandGuidelineAlignmentResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - api_start_time
        logger.info(f"[{agent_key}] Received response from Gemini. API call took {api_duration:.2f} seconds.")
        
        result_text = response.text
        if not result_text:
            raise ValueError("Empty response from Gemini")
            
        clean_json = _clean_json_string(result_text)
        result_data = json.loads(clean_json)
        result_data["status"] = "completed"
        
        parsed_result = BrandGuidelineAlignmentResult(**result_data)

        # 5. Save results to Database
        # Ideation output
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": project_id},
            {"$set": {
                "brand_guardrails": parsed_result.brand_guardrails.model_dump() if parsed_result.brand_guardrails else None,
                f"status.{agent_key}": "completed"
            }},
            upsert=True
        )
        
        # Pipeline log output
        duration = time.time() - start_time
        pipeline_log_entry = {
            "agent_key": agent_key,
            "status": "completed",
            "reasoning": parsed_result.reasoning,
            "mandatory_translation_log": (
                [m.model_dump() for m in parsed_result.mandatory_translation_log] 
                if parsed_result.mandatory_translation_log else []
            ),
            "cultural_sensitivity_notes": parsed_result.cultural_sensitivity_notes,
            "duration": round(duration, 3),
            "updated_at": time.time()
        }
        
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": project_id},
            {"$push": {"logs": pipeline_log_entry}},
            upsert=True
        )

        logger.info(f"[{agent_key}] Completed agent execution in {duration:.2f}s")
        return parsed_result.model_dump()

    except Exception as e:
        logger.error(f"[{agent_key}] Error: {str(e)}")
        error_result = BrandGuidelineAlignmentResult(
            status="error",
            reason=str(e)
        )
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": project_id},
            {"$set": {
                f"status.{agent_key}": "error",
                f"error.{agent_key}": str(e)
            }},
            upsert=True
        )
        return error_result.model_dump()
