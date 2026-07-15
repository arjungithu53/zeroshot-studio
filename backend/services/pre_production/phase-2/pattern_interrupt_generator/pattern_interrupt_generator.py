import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")

logger = logging.getLogger("zeroshot.phase2.pattern_interrupt_generator")

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _clean_json_string(raw_text: str) -> str:
    """Strips markdown code fences like ```json ... ```."""
    stripped = raw_text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json"):]
    elif stripped.startswith("```"):
        stripped = stripped[len("```"):]
    
    if stripped.endswith("```"):
        stripped = stripped[:-3]
        
    return stripped.strip()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class InversionItem(BaseModel):
    convention: str = Field(..., description="The dominant executional convention in the product category.")
    inversion: str = Field(..., description="The 180-degree opposite execution that is perfectly wrong but still answers the brief.")
    product_grounded_reason: str = Field(..., description="How this inversion shows something true about the product.")

class LensOutput(BaseModel):
    lens: str = Field(..., description="The concepting lens applied (e.g. Invert, Exaggerate, Eliminate, Symbolize, Use the medium, Metaphor).")
    hook_seed: str = Field(..., description="The generated pattern interrupt hook seed.")

class Agent16Result(BaseModel):
    reasoning: str = Field(..., description="Reasoning behind the generated hook seeds and inversions.")
    inversion_table: List[InversionItem] = Field(..., description="Table of inverted conventions.")
    concepting_lens_outputs: List[LensOutput] = Field(..., description="Outputs from running the SMP through concepting lenses.")
    seed_list: List[str] = Field(..., description="Final list of 5-8 pattern interrupt hook seeds, to be injected into Agent 20.")
    status: Optional[str] = Field(None, description="Status of the agent execution.")

# ---------------------------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------------------------
async def run_pattern_interrupt_generator_agent(project_id: str, db: Any) -> Agent16Result:
    """
    Agent 16: Pattern Interrupt Generator
    Generates 5-8 hook seeds by going 180° against category conventions, always with a product-grounded reason.
    """
    logger.info(f"Initializing Agent 16 (Pattern Interrupt Generator) for project_id={project_id}...")
    start_time = time.time()

    if not GEMINI_API_KEY:
        logger.error("Agent 16: GEMINI_API_KEY environment variable not set.")
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    # 1. Fetch data
    logger.info(f"Agent 16: Fetching data for project_id={project_id}...")
    
    # Project Document
    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
    if not project_doc:
        raise ValueError(f"Project '{project_id}' not found in {PROJECTS_COLLECTION}.")

    # Strategy Document
    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
    if not strategy_doc:
        raise ValueError(f"No STRATEGY document found for project '{project_id}'.")

    # Ideation Document
    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
    if not ideation_doc:
        raise ValueError(f"No IDEATION document found for project '{project_id}'.")

    # Extract required inputs
    # From strategy
    campaign_platform = strategy_doc.get("campaign_platform", "Unknown")
    human_truth = strategy_doc.get("human_truth", "Unknown")
    enemy = strategy_doc.get("enemy", "Unknown")
    
    creative_brief = strategy_doc.get("creative_brief", {})
    smp = creative_brief.get("single_minded_proposition", "Unknown")
    
    competitive_landscape = strategy_doc.get("competitive_landscape", {})
    conventions_to_break = competitive_landscape.get("conventions_to_break", [])
    repeated_messages = competitive_landscape.get("repeated_messages", [])
    
    venn_model = strategy_doc.get("strategy_models", {}).get("venn_model", {})
    competitor_gap = venn_model.get("competitor_gap", [])
    
    # From ideation
    video_type_final = ideation_doc.get("video_type_final", "Unknown")
    brand_guardrails = ideation_doc.get("brand_guardrails", {})
    format_group = ideation_doc.get("format_group", "N")
    is_visual = format_group == "V"

    # For Group V: extract ingredient context from visual_structure
    visual_structure = ideation_doc.get("visual_structure", {})
    texture_inventory = visual_structure.get("texture_inventory", [])
    product_details = project_doc.get("product_details", "")

    logger.info(f"Agent 16: Extracted inputs - SMP='{smp}', Video Type='{video_type_final}', format_group={format_group}, enemy='{enemy}'.")

    # 2. Construct Prompt — format-conditional
    if is_visual:
        prompt = f"""You are the Pattern Interrupt Generator Agent for a {video_type_final} advertisement.
Purpose: Generate 5-8 VISUALLY EXECUTABLE hook seeds that are impossible to scroll past at frame 0.

For a {video_type_final} format, pattern interrupts are NOT narrative provocations — they are VISUAL surprises grounded in the product's actual physical ingredients and material properties. The Bernbach principle still applies: every inversion must reveal something true about the product's real ingredients.

PRODUCT:
{product_details}

VISUAL STRUCTURE (ingredients and textures already identified):
- Texture inventory: {json.dumps(texture_inventory)}

STRATEGY CONTEXT (for brand grounding only — do not translate into narrative hooks):
- Single Minded Proposition: {smp}
- Brand Guardrails: {json.dumps(brand_guardrails)}
- Competitor visual conventions to break: {json.dumps(conventions_to_break)}
- Repeated category messages to avoid: {json.dumps(repeated_messages)}

VISUAL CATEGORY CONVENTIONS (what competitors always do visually):
{json.dumps(conventions_to_break)}

PROMPT LOGIC:

Step 1: Extract the hero ingredients from the product description. For each ingredient, identify:
- Its specific color at macro scale (exact hue, not generic "yellow")
- Its material state (powder, liquid, crystalline, gel, oil, solid)
- Its most surprising or least-seen visual property (what does no one ever show?)
- What it looks like under extreme macro and controlled lighting

Step 2: Identify 3 dominant VISUAL conventions in this product category from conventions_to_break. These are the visual norms to invert — e.g. "always slow-mo water drop", "always white minimalist background", "always show skin transformation".

Step 3: Apply the Rong inversion to each visual convention. For each norm:
- What is the 180-degree opposite visual execution that is perfectly wrong but still shows something true about the product's ingredients?
- Cross-check: does this inversion showcase a real material property of a real ingredient? If not, rewrite it.
- Cross-check against repeated_messages: any seed that echoes a repeated message must be rewritten.

Step 4: Apply the concepting lens table to the product's INGREDIENT PROPERTIES (not the SMP narrative). Run the product's most distinctive ingredient through at least 5 lenses:
- Invert: show the ingredient at an unexpected scale, state, or angle
- Exaggerate: push one visual property to an absurd but truthful extreme
- Eliminate: remove everything except the single most visually arresting property
- Symbolize: let the ingredient's color or texture stand in for the entire product promise
- Use the medium: exploit what CGI/3D can do that live action cannot (particle physics, material morph, non-literal scale)
- Metaphor: let the ingredient's material behavior become a visual metaphor for the product's effect

Each seed must be:
1. Visually specific — describable in one camera direction sentence
2. Ingredient-grounded — references a real physical material from the product
3. Executable in {video_type_final} — achievable as CGI/3D, no human talent required

Output must exactly match the JSON schema provided. Return 5-8 visual hook seeds with source lens labelled in concepting_lens_outputs, and a flat seed_list.
"""
    else:
        prompt = f"""You are the Pattern Interrupt Generator Agent .
Purpose: Grounded in 'The Art of Being Rong'.
Generate 5-8 hook seeds by going 180 degrees against category conventions — always with a product-grounded reason (Bernbach principle: the inversion must show something true about the product).

INPUT DATA:
- Campaign Platform: {campaign_platform}
- Human Truth: {human_truth}
- Enemy: {enemy}
- Single Minded Proposition (SMP): {smp}
- Conventions to Break: {json.dumps(conventions_to_break)}
- Competitor Gap: {json.dumps(competitor_gap)}
- Repeated Messages: {json.dumps(repeated_messages)}

FROM IDEATION:
- Video Type Final: {video_type_final}
- Brand Guardrails: {json.dumps(brand_guardrails)}

PROMPT LOGIC:
Step 1: Identify 3 dominant executional conventions in this product category. Draw from conventions_to_break and competitor_gap. These are the norms to invert.

Step 2: Apply the Rong inversion technique to each convention.
- Cross-reference with competitor_gap and repeated_messages. A pattern interrupt is strongest when it hits a competitor_gap rather than merely inverting a generic norm.
- Cross-check all generated seeds against repeated_messages. Any seed that uses language appearing in repeated_messages must be rewritten.
- For each norm ask: what is the 180-degree opposite execution that is perfectly wrong but still answers the brief?
- Ground the Rong idea in the "Enemy".

Step 3: Apply the concepting lens table.
Run the Single Minded Proposition through at least 5 of the following lenses: Invert, Exaggerate, Eliminate, Symbolize, Use the medium, Metaphor. Produce one hook seed per lens.

Output must exactly match the JSON schema provided. Return 5-8 pattern interrupt hook seeds with source lens labelled in concepting_lens_outputs, and a flat seed_list of the 5-8 hook seeds.
"""

    # 3. Call Gemini API
    invoke_start = time.time()
    logger.info(f"Agent 16: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 16: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": Agent16Result.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 16: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 16: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 16: Successfully parsed JSON response.")

        result = Agent16Result(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 16: Successfully validated structured output with Pydantic.")

    except Exception as e:
        logger.error(f"Agent 16: Gemini API Call failed: {e}")
        failure_log = {
            "agent_id": "agent_16_pattern_interrupt_generator",
            "status": "failed",
            "timestamp": time.time(),
            "error": str(e)
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": failure_log}},
            upsert=True
        )
        raise ValueError(f"Agent 16 execution failed: {str(e)}")

    # 4. DB Updates
    logger.info(f"Agent 16: Updating PIPELINE and IDEATION collections...")

    # Create pipeline log
    pipeline_log = {
        "agent_id": "agent_16_pattern_interrupt_generator",
        "timestamp": time.time(),
        "execution_time_seconds": round(time.time() - start_time, 2),
        "status": "completed",
        "reasoning": result.reasoning,
        "inversion_table": [item.model_dump() for item in result.inversion_table],
        "concepting_lens_outputs": [item.model_dump() for item in result.concepting_lens_outputs],
        "seed_list": result.seed_list
    }

    try:
        # Push to PIPELINE_COLLECTION
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )

        # Output logic for IDEATION: No specific fields mentioned for IDEATION in output schema, just state.
        # But we'll save the result into ideation under pattern_interrupt_generator for good measure.
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "pattern_interrupt_generator": {
                    "reasoning": result.reasoning,
                    "inversion_table": pipeline_log["inversion_table"],
                    "concepting_lens_outputs": pipeline_log["concepting_lens_outputs"],
                    "seed_list": result.seed_list
                },
                "updated_at": time.time()
            }}
        )
        logger.info(f"Agent 16: DB updates successful.")
    except Exception as e:
        logger.error(f"Agent 16: Error updating database: {e}")
        raise ValueError(f"Agent 16 DB update failed: {str(e)}")

    total_duration = time.time() - start_time
    logger.info(f"Agent 16: Completed successfully in {total_duration:.2f}s")

    return result
