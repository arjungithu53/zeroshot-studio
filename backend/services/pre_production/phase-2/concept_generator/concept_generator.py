import os
import time
import json
import logging
from typing import Any, List, Optional
from bson import ObjectId
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")

logger = logging.getLogger("zeroshot.phase2.concept_generator")


# ---------------------------------------------------------------------------
# Format-group schemas
# ---------------------------------------------------------------------------

class NarrativeConceptSpec(BaseModel):
    concept_id: str = Field(description="Unique identifier (e.g. Concept_A). Must match the slot_id it fills.")
    concept_hook: str = Field(description="One sentence functioning as both a summary and a scroll-stopper. Must reflect the slot's hook_mechanism_type.")
    story_beats: List[str] = Field(description="2-3 beats maximum. Structural only, no script. Maps to assigned narrative_skeleton variant.")
    archetype: str = Field(description="The narrative archetype this concept executes (from selected_archetype).")
    virality_lever: str = Field(description="Which specific virality mechanic is embedded and at which beat it activates.")
    constraint_anchors: List[str] = Field(description="List of hard constraints this concept is complying with.")
    offer_placement: Optional[str] = Field(None, description="Populated later by Agent 21a.")
    cta_framing: str = Field(description="The call-to-action language register (e.g. intimate whisper, defiant declaration).")
    character_description: str = Field(description="Specific description of the character(s) who appear — age range, appearance, energy, wardrobe. Required for Veo character consistency. Use 'No human talent' if format excludes people.")
    category: Optional[str] = Field(None, description="Populated later by Agent 22.")
    kill_score: Optional[float] = Field(None, description="Populated later by Agent 25.")
    provenance: str = Field(description="Which specific strategy inputs and diversity slot this concept is built from.")
    risk_tags: List[str] = Field(description="Any potential execution risks, cultural sensitivities, or platform survivability concerns.")


class VisualConceptSpec(BaseModel):
    concept_id: str = Field(description="Unique identifier (e.g. Concept_A). Must match the slot_id it fills.")
    concept_hook: str = Field(description="One sentence describing the opening visual event — specific, arresting, format-native. Must reflect the slot's hook_mechanism_type.")
    composition_beats: List[str] = Field(description="2-4 visual beats describing how the frame evolves — texture change, angle shift, lighting reveal, product interaction. NO character arc. NO dialogue structure. Visual flow only.")
    visual_motif: str = Field(description="The organizing visual motif this concept executes (from selected_visual_motif).")
    texture_moments: List[str] = Field(description="2-3 specific texture or material interactions that appear in this concept.")
    motion_choreography: str = Field(description="The primary camera and product movement pattern for this concept.")
    lighting_rationale: str = Field(description="How lighting serves this concept's visual objective.")
    virality_lever: str = Field(description="Which specific virality mechanic is embedded (e.g. ASMR texture reveal, unexpected scale, transformation loop).")
    constraint_anchors: List[str] = Field(description="List of hard constraints this concept is complying with.")
    offer_placement: Optional[str] = Field(None, description="Populated later by Agent 21a.")
    cta_framing: str = Field(description="The call-to-action delivery mechanism (e.g. text super, VO whisper over macro shot).")
    category: Optional[str] = Field(None, description="Populated later by Agent 22.")
    kill_score: Optional[float] = Field(None, description="Populated later by Agent 25.")
    provenance: str = Field(description="Which specific visual structure inputs and diversity slot this concept is built from.")
    risk_tags: List[str] = Field(description="Any potential execution risks or AI video generation concerns.")


class GenerationLogEntry(BaseModel):
    concept_id: str
    slot_id: str
    hook_mechanism_used: str


class OrthogonalityCheck(BaseModel):
    hook_type_spread: List[str]
    emotional_axis_spread: List[str]
    payoff_spread: List[str]
    cluster_warnings: List[str]


class NarrativeConceptGeneratorResult(BaseModel):
    concept_portfolio: List[NarrativeConceptSpec] = Field(description="Array of 10 narrative concept specifications, one per diversity slot.")
    reasoning: str
    concept_generation_log: List[GenerationLogEntry]
    orthogonality_check: OrthogonalityCheck
    status: Optional[str] = Field(default="completed")


class VisualConceptGeneratorResult(BaseModel):
    concept_portfolio: List[VisualConceptSpec] = Field(description="Array of 10 visual concept specifications, one per diversity slot.")
    reasoning: str
    concept_generation_log: List[GenerationLogEntry]
    orthogonality_check: OrthogonalityCheck
    status: Optional[str] = Field(default="completed")


def _clean_json_string(json_str: str) -> str:
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    return json_str.strip()


async def run_concept_generator_agent(project_id: str, db: Any):
    logger.info(f"Initializing Agent 20 (Concept Generator) for project_id={project_id}...")

    try:
        project_obj_id = ObjectId(project_id)
    except Exception as e:
        raise ValueError(f"Invalid project_id {project_id}")

    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": project_obj_id})
    if not project_doc:
        raise ValueError(f"Project not found in {PROJECTS_COLLECTION}")

    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}

    # ---------------------------------------------------------------------------
    # Format group routing
    # ---------------------------------------------------------------------------
    format_group = ideation_doc.get("format_group", "N")
    video_type_final = ideation_doc.get("video_type_final", "UGC")
    logger.info(f"Agent 20: format_group={format_group} video_type={video_type_final}")

    # ---------------------------------------------------------------------------
    # Strategy extracts
    # ---------------------------------------------------------------------------
    strat_agents = strategy_doc.get("agents", {})
    venn_model = strat_agents.get("strategy_models", {}).get("venn_model", {})
    comp_landscape = strat_agents.get("competitive_landscape", {})
    creative_brief = strat_agents.get("insight_validation", {}).get("creative_brief", {})

    project_details = {
        "product_details": project_doc.get("product_details", ""),
        "product_url": project_doc.get("product_url") or "Not provided",
        "location_context": strat_agents.get("audience_persona", {}).get("location_context", ""),
        "human_truth": strat_agents.get("central_human_truth", {}).get("human_truth", ""),
        "enemy": strat_agents.get("conflict_identification", {}).get("enemy", ""),
        "campaign_platform": strat_agents.get("truth_conflict_platform", {}).get("selected_platform", ""),
        "creative_brief": creative_brief,
        "competitive_whitespace": strat_agents.get("positioning_alignment", {}).get("competitive_whitespace", ""),
        "conventions_to_follow": comp_landscape.get("conventions_to_follow", []),
        "brand_can_say": venn_model.get("brand_can_say", []),
        "audience_cares_about": venn_model.get("audience_cares_about", []),
        "competitor_gap": venn_model.get("competitor_gap", []),
    }

    # ---------------------------------------------------------------------------
    # Ideation architecture — format-conditional
    # ---------------------------------------------------------------------------
    architecture = {
        "constraint_graph": ideation_doc.get("constraint_graph", {}),
        "priority_directives": ideation_doc.get("priority_directives", {}),
        "idea_core_rules": ideation_doc.get("idea_core_rules", {}),
        "brand_guardrails": ideation_doc.get("brand_guardrails", {}),
        "video_type_final": video_type_final,
        "format_group": format_group,
        "video_type_conditioning_notes": ideation_doc.get("video_type_conditioning_notes", ""),
        "narrative_budget": ideation_doc.get("narrative_budget", {}),
        "platform_rules": ideation_doc.get("platform_rules", {}),
    }

    if format_group == "N":
        architecture["scene_intelligence"] = ideation_doc.get("scene_intelligence", {})
        architecture["scene_integration_plan"] = ideation_doc.get("scene_integration_plan", {})
        architecture["narrative_skeleton"] = ideation_doc.get("narrative_skeleton", {})
        architecture["narrative_plan"] = ideation_doc.get("narrative_plan", {})
        architecture["selected_archetype"] = ideation_doc.get("selected_archetype", {})
    else:
        architecture["visual_structure"] = ideation_doc.get("visual_structure", {})
        architecture["selected_visual_motif"] = ideation_doc.get("selected_visual_motif", {})
        architecture["scene_intelligence"] = ideation_doc.get("scene_intelligence", {})

    # ---------------------------------------------------------------------------
    # Diversity manifest — the generation grid
    # ---------------------------------------------------------------------------
    diversity_manifest = ideation_doc.get("diversity_manifest", {})
    concept_slots = diversity_manifest.get("concept_slots", [])
    if not concept_slots:
        logger.warning("Agent 20: diversity_manifest is empty — generating without slot constraints")

    # ---------------------------------------------------------------------------
    # Creative seeds — typed field retrieval (not string-match on pipeline_logs)
    # ---------------------------------------------------------------------------
    intergalactic_seeds = ideation_doc.get("intergalactic_seeds", [])
    pattern_interrupt_data = ideation_doc.get("pattern_interrupt_generator", {})
    pattern_interrupt_seeds = pattern_interrupt_data.get("seed_list", [])

    # Virality directive — still read from pipeline logs as it is stored by viral_mechanics agent
    virality_directive = ""
    mental_model_seeds = []
    pipeline_doc = await db[PIPELINE_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    for log in pipeline_doc.get("agent_logs", []):
        name = str(log.get("agent_name", ""))
        agent_id = str(log.get("agent_id", ""))
        payload = log.get("output") or log.get("data") or log
        if "viral_mechanics" in name or "viral_mechanics" in agent_id:
            virality_directive = payload.get("virality_directive", "")
        elif "mental_model" in name or "mental_model" in agent_id:
            mental_model_seeds = payload.get("seed_list", [])

    logger.info(f"Agent 20: Extracted all inputs | intergalactic_seeds={len(intergalactic_seeds)} pattern_interrupt_seeds={len(pattern_interrupt_seeds)} concept_slots={len(concept_slots)}")

    # ---------------------------------------------------------------------------
    # Build pinned creative brief — injected at top of prompt for brief fidelity
    # ---------------------------------------------------------------------------
    persona = strat_agents.get("audience_persona", {})
    pinned_brief = {
        "product": project_doc.get("product_details", ""),
        "brand_adjective": strat_agents.get("brand_adjective", ""),
        "human_truth": strat_agents.get("central_human_truth", {}).get("human_truth", ""),
        "enemy": strat_agents.get("conflict_identification", {}).get("enemy", ""),
        "enemy_type": strat_agents.get("conflict_identification", {}).get("enemy_type", ""),
        "single_minded_proposition": creative_brief.get("single_minded_proposition", ""),
        "persona_name": persona.get("persona_name", ""),
        "persona_age_range": persona.get("age_range", ""),
        "persona_location": persona.get("location_context", ""),
        "persona_core_motivation": persona.get("core_motivation", ""),
        "persona_core_fear": persona.get("core_fear", ""),
        "persona_buying_driver": persona.get("buying_driver", ""),
        "positioning_statement": strat_agents.get("positioning_alignment", {}).get("positioning_statement", ""),
        "brand_can_say": venn_model.get("brand_can_say", []),
        "offer": project_doc.get("price_and_offer", ""),
    }

    # ---------------------------------------------------------------------------
    # Build format-specific prompt and schema
    # ---------------------------------------------------------------------------
    if format_group == "V":
        format_instructions = f"""
You are generating VISUAL concepts for a {video_type_final} advertisement.
This format is VISUAL-FIRST. There are no characters, no story arcs, no dialogue.

CRITICAL RULES FOR GROUP V CONCEPTS:
- composition_beats describes how the VISUAL FRAME evolves — texture change, angle shift, lighting reveal, product interaction
- composition_beats must NOT describe character actions, emotional reactions, or narrative progression
- NO character arc. NO dialogue structure. NO human talent (unless format explicitly permits)
- The hook is a VISUAL EVENT — a specific, arresting frame — not a rhetorical question or personal statement
- visual_motif from selected_visual_motif must be honored in every concept
- Every concept must specify its texture_moments, motion_choreography, and lighting_rationale
- If the format is Product Beauty or Flatlay: character_description equivalent is not applicable — do not invent characters

VISUAL ARCHITECTURE:
{json.dumps(architecture.get("visual_structure", {}), indent=2)}

SELECTED VISUAL MOTIF:
{json.dumps(architecture.get("selected_visual_motif", {}), indent=2)}
"""
        output_schema_class = VisualConceptGeneratorResult
        concept_spec_name = "VisualConceptSpec"
    else:
        format_instructions = f"""
You are generating NARRATIVE concepts for a {video_type_final} advertisement.
This format is NARRATIVE-FIRST. story_beats, archetype, and character_description are required.

CRITICAL RULES FOR GROUP N CONCEPTS:
- story_beats describes the NARRATIVE ARC — not visual shots
- archetype must match the selected_archetype and drive the emotional logic
- character_description must be specific enough for Veo character consistency injection
- hook_mechanism_type from the slot governs the hook physics — honor it exactly
- For UGC: tone must feel unscripted, creator-native, not brand-polished
- For Testimonial: at least one beat must contain a specific result claim
- For Satire: the enemy must be personified and the brand must be the straight-faced solution

NARRATIVE ARCHITECTURE:
Narrative skeleton: {json.dumps(architecture.get("narrative_skeleton", {}), indent=2)}
Narrative plan: {json.dumps(architecture.get("narrative_plan", {}), indent=2)}
Selected archetype: {json.dumps(architecture.get("selected_archetype", {}), indent=2)}
"""
        output_schema_class = NarrativeConceptGeneratorResult
        concept_spec_name = "NarrativeConceptSpec"

    prompt = f"""You are the Concept Generator Agent (Agent 20).
Your purpose: Fill the 10 ConceptSlots in the DiversityManifest with structurally distinct concept specifications.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CREATIVE BRIEF — READ THIS BEFORE GENERATING ANYTHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is the non-negotiable strategic foundation. Every concept must be traceable back to this brief. A concept that could apply to any product in this category has failed.

Product: {pinned_brief["product"]}
Brand adjective: {pinned_brief["brand_adjective"]}
Offer: {pinned_brief["offer"]}

Persona: {pinned_brief["persona_name"]}, {pinned_brief["persona_age_range"]}, {pinned_brief["persona_location"]}
Core motivation: {pinned_brief["persona_core_motivation"]}
Core fear: {pinned_brief["persona_core_fear"]}
Buying driver: {pinned_brief["persona_buying_driver"]}

Human truth: {pinned_brief["human_truth"]}
Enemy: {pinned_brief["enemy"]} ({pinned_brief["enemy_type"]})
Single-minded proposition: {pinned_brief["single_minded_proposition"]}
Positioning: {pinned_brief["positioning_statement"]}

What the brand can credibly claim: {json.dumps(pinned_brief["brand_can_say"])}

BRIEF FIDELITY TEST (apply to every concept before finalizing):
Ask yourself: could this concept be used by any other brand in this category? If yes — it has failed. Every concept must reference something specific to THIS product (its ingredients, its texture, its formula, its certification) AND something specific to THIS persona (her actual life situation, her specific fear, her buying trigger). Generic concepts that could apply to any moisturiser are rejected.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GENERATION RULE: Generate exactly one concept per slot. The concept_id must match the slot_id. The concept's hook must be driven by the slot's hook_mechanism_type.

{format_instructions}

DIVERSITY MANIFEST (your generation grid — one concept per slot, no deviations):
{json.dumps(concept_slots, indent=2)}

COMPLIANCE CHECKS:
- Must satisfy at least 2 conventions_to_follow
- Must be supportable by brand_can_say whitelist
- Must map to audience_cares_about
- At least 2 concepts MUST exploit competitor_gap
- Embed virality_directive in every concept
- At least 1 concept must use an intergalactic_seed as its hook foundation
- At least 2 concepts must be driven by a pattern_interrupt_seed

ARCHITECTURE INPUTS:
{json.dumps(architecture, indent=2)}

STRATEGY & PROJECT DATA:
{json.dumps(project_details, indent=2)}

CREATIVE SEEDS:
Pattern Interrupt Seeds: {json.dumps(pattern_interrupt_seeds, indent=2)}
Mental Model Seeds: {json.dumps(mental_model_seeds, indent=2)}
Intergalactic Seeds: {json.dumps(intergalactic_seeds, indent=2)}
Virality Directive: "{virality_directive}"

Generate exactly 10 concepts in concept_portfolio. Before submitting, apply the BRIEF FIDELITY TEST above to each concept. Return valid JSON matching the schema.
"""

    invoke_start = time.time()
    logger.info(f"Agent 20: Calling Gemini model={GEMINI_MODEL} format_group={format_group}")

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": output_schema_class.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 20: Gemini completed | duration={api_duration:.2f}s")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        result = output_schema_class(**parsed_data)
        result.status = "completed"

    except Exception as e:
        logger.error(f"Agent 20: Gemini inference failed: {str(e)}")
        raise

    # ---------------------------------------------------------------------------
    # Write to DB — unified dict format for downstream compatibility
    # ---------------------------------------------------------------------------
    portfolio_dict = [c.model_dump() for c in result.concept_portfolio]

    pipeline_log_output = {
        "reasoning": result.reasoning,
        "format_group": format_group,
        "concept_generation_log": [entry.model_dump() for entry in result.concept_generation_log],
        "orthogonality_check": result.orthogonality_check.model_dump(),
    }

    await db[IDEATION_COLLECTION].update_one(
        {"project_id": str(project_id)},
        {"$set": {"concept_portfolio": portfolio_dict}},
        upsert=True,
    )

    pipeline_log = {
        "agent_id": 20,
        "agent_name": "concept_generator",
        "format_group": format_group,
        "execution_time_sec": round(time.time() - invoke_start, 2),
        "timestamp": time.time(),
        "status": "completed",
        "output": pipeline_log_output,
    }
    await db[PIPELINE_COLLECTION].update_one(
        {"project_id": str(project_id)},
        {"$push": {"agent_logs": pipeline_log}},
        upsert=True,
    )

    total_duration = time.time() - invoke_start
    logger.info(f"Agent 20: Completed in {total_duration:.2f}s | {len(result.concept_portfolio)} concepts | format_group={format_group}")
    return result
