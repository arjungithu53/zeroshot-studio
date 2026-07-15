import os
import sys
import logging
import time
from datetime import datetime
from typing import TypedDict

# Register all agent directories on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
for _agent_dir in [
    "",
    "creative_constraint_manager",
    "constraint_priority_resolver",
    "idea_core_preservation",
    "brand_guideline_alignment",
    "video_type_selection_agent",
    "duration_structuring_agent",
    "scene_deconstruction_agent",
    "scene_role_enumeration_agent",
    "scene_role_selector_agent",
    "temporal_placement_solver_agent",
    "scene_integration_plan",
    "narrative_skeleton_generator",
    "narrative_skeleton_planner",
    "narrative_archetype_selector",
    "visual_structure_agent",
    "visual_motif_selector",
    "platform_behavior_optimizer",
    "pattern_interrupt_generator",
    "viral_mechanics_agent",
    "mental_model_transformer",
    "intergalactic_thinking_agent",
    "diversity_manifest_generator",
    "concept_generator",
    "phase2_concept_reviewer",
    "offer_narrative_integrator",
    "concept_categorization_agent",
    "interest_filter_agent",
    "concept_kill_switch",
    "concept_mutation_agent",
]:
    _path = os.path.join(_HERE, _agent_dir) if _agent_dir else _HERE
    if _path not in sys.path:
        sys.path.insert(0, _path)

from langgraph.graph import StateGraph, START, END

from creative_constraint_manager import run_creative_constraint_manager_agent
from constraint_priority_resolver import run_constraint_priority_resolver_agent
from idea_core_preservation import run_idea_core_preservation_agent
from brand_guideline_alignment import run_brand_guideline_alignment_agent
from video_type_conditioning_agent import run_video_type_conditioning_agent
from video_type_selection_agent import run_video_type_selection_agent
from duration_structuring_agent import run_duration_structuring_agent
from scene_deconstruction_agent import run_scene_deconstruction_agent
from scene_role_enumeration_agent import run_scene_role_enumeration_agent
from scene_role_selector_agent import run_scene_role_selector_agent
from temporal_placement_solver_agent import run_temporal_placement_solver_agent
from scene_integration_plan import run_scene_integration_plan_agent
from narrative_skeleton_generator import run_narrative_skeleton_generator
from narrative_skeleton_planner import run_narrative_skeleton_planner
from narrative_archetype_selector import run_narrative_archetype_selector
from visual_structure_agent import run_visual_structure_agent
from visual_motif_selector import run_visual_motif_selector
from platform_behavior_optimizer import run_platform_behavior_optimizer_agent
from pattern_interrupt_generator import run_pattern_interrupt_generator_agent
from viral_mechanics_agent import run_viral_mechanics_agent
from mental_model_transformer import run_mental_model_transformer
from intergalactic_thinking_agent import run_intergalactic_thinking_agent
from diversity_manifest_generator import run_diversity_manifest_generator
from concept_generator import run_concept_generator_agent
from phase2_concept_reviewer import run_phase2_concept_reviewer
from offer_narrative_integrator import run_offer_narrative_integrator_agent
from concept_categorization_agent import run_concept_categorization_agent
from interest_filter_agent import run_interest_filter_agent
from concept_kill_switch import run_concept_kill_switch_agent
from concept_mutation_agent import run_concept_mutation_agent

logger = logging.getLogger("zeroshot.phase2.orchestrator")

# ---------------------------------------------------------------------------
# Format group classification (canonical set — case-insensitive match)
# ---------------------------------------------------------------------------
_VISUAL_FORMATS = {
    "product beauty",
    "flatlay",
    "cgi/3d product",
    "cgi3d product",
    "animated",
    "animation / illustrated",
    "animation/illustrated",
    "cgi/3d",
    "cgi 3d",
}

_db = None


class State(TypedDict):
    project_id: str
    format_group: str        # "N" or "V", set by format_group_detector
    regen_count: int         # number of mutation rounds completed (starts at 0)
    kill_switch_status: str  # "approved" | "needs_regen"


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------

async def node_creative_constraint_manager(state: State) -> dict:
    await run_creative_constraint_manager_agent(state["project_id"], _db)
    return {}

async def node_constraint_priority_resolver(state: State) -> dict:
    await run_constraint_priority_resolver_agent(state["project_id"], _db)
    return {}

async def node_idea_core_preservation(state: State) -> dict:
    await run_idea_core_preservation_agent(state["project_id"], _db)
    return {}

async def node_brand_guideline_alignment(state: State) -> dict:
    await run_brand_guideline_alignment_agent(state["project_id"], _db)
    return {}

async def node_video_type_conditioning(state: State) -> dict:
    await run_video_type_conditioning_agent(state["project_id"], _db)
    return {}

async def node_video_type_selection(state: State) -> dict:
    await run_video_type_selection_agent(state["project_id"], _db)
    return {}

async def node_format_group_detector(state: State) -> dict:
    """
    Reads video_type_final from the ideation document, classifies it as Group N or V,
    writes format_group to the ideation document, and returns it into LangGraph state.
    This is the routing decision point for Phase 2-N vs Phase 2-V.
    """
    project_id = state["project_id"]
    ideation_doc = await _db["ideation"].find_one({"project_id": str(project_id)}) or {}
    video_type_final = ideation_doc.get("video_type_final", "").lower().strip()
    format_group = "V" if video_type_final in _VISUAL_FORMATS else "N"
    logger.info("format_group_detector | video_type=%s format_group=%s | project_id=%s",
                video_type_final, format_group, project_id)
    await _db["ideation"].update_one(
        {"project_id": str(project_id)},
        {"$set": {"format_group": format_group}},
        upsert=True,
    )
    return {"format_group": format_group}

async def node_duration_structuring(state: State) -> dict:
    await run_duration_structuring_agent(state["project_id"], _db)
    return {}

async def node_scene_deconstruction(state: State) -> dict:
    await run_scene_deconstruction_agent(state["project_id"], _db)
    return {}

# ── Phase 2-N nodes ──────────────────────────────────────────────────────────

async def node_scene_role_enumeration(state: State) -> dict:
    await run_scene_role_enumeration_agent(state["project_id"], _db)
    return {}

async def node_scene_role_selector(state: State) -> dict:
    await run_scene_role_selector_agent(state["project_id"], _db)
    return {}

async def node_temporal_placement_solver(state: State) -> dict:
    await run_temporal_placement_solver_agent(state["project_id"], _db)
    return {}

async def node_scene_integration_plan(state: State) -> dict:
    await run_scene_integration_plan_agent(state["project_id"], _db)
    return {}

async def node_narrative_skeleton_generator(state: State) -> dict:
    await run_narrative_skeleton_generator(state["project_id"], _db)
    return {}

async def node_narrative_skeleton_planner(state: State) -> dict:
    await run_narrative_skeleton_planner(state["project_id"], _db)
    return {}

async def node_narrative_archetype_selector(state: State) -> dict:
    await run_narrative_archetype_selector(state["project_id"], _db)
    return {}

# ── Phase 2-V nodes ──────────────────────────────────────────────────────────

async def node_visual_structure_agent(state: State) -> dict:
    await run_visual_structure_agent(state["project_id"], _db)
    return {}

async def node_visual_motif_selector(state: State) -> dict:
    await run_visual_motif_selector(state["project_id"], _db)
    return {}

# ── Shared continuation nodes ────────────────────────────────────────────────

async def node_platform_behavior_optimizer(state: State) -> dict:
    await run_platform_behavior_optimizer_agent(state["project_id"], _db)
    return {}

async def node_pattern_interrupt_generator(state: State) -> dict:
    await run_pattern_interrupt_generator_agent(state["project_id"], _db)
    return {}

async def node_viral_mechanics(state: State) -> dict:
    await run_viral_mechanics_agent(state["project_id"], _db)
    return {}

async def node_mental_model_transformer(state: State) -> dict:
    await run_mental_model_transformer(state["project_id"], _db)
    return {}

async def node_intergalactic_thinking(state: State) -> dict:
    await run_intergalactic_thinking_agent(state["project_id"], _db)
    return {}

async def node_diversity_manifest_generator(state: State) -> dict:
    await run_diversity_manifest_generator(state["project_id"], _db)
    return {}

async def node_concept_generator(state: State) -> dict:
    await run_concept_generator_agent(state["project_id"], _db)
    return {}

async def node_phase2_concept_reviewer(state: State) -> dict:
    await run_phase2_concept_reviewer(state["project_id"], _db)
    return {}

async def node_offer_narrative_integrator(state: State) -> dict:
    await run_offer_narrative_integrator_agent(state["project_id"], _db)
    return {}

async def node_concept_categorization(state: State) -> dict:
    await run_concept_categorization_agent(state["project_id"], _db)
    return {}

async def node_interest_filter(state: State) -> dict:
    await run_interest_filter_agent(state["project_id"], _db)
    return {}

async def node_concept_kill_switch(state: State) -> dict:
    await run_concept_kill_switch_agent(state["project_id"], _db)
    # Read ideation to determine routing — approved_concepts populated means approved
    doc = await _db["ideation"].find_one({"project_id": state["project_id"]}) or {}
    approved = doc.get("phase_2_output", {}).get("approved_concepts", [])
    status = "approved" if approved else "needs_regen"
    logger.info("concept_kill_switch: status=%s | project_id=%s", status, state["project_id"])
    return {"kill_switch_status": status}


async def node_concept_mutation(state: State) -> dict:
    await run_concept_mutation_agent(state["project_id"], _db)
    new_count = state.get("regen_count", 0) + 1
    logger.info("concept_mutation: completed round %d | project_id=%s", new_count, state["project_id"])
    return {"regen_count": new_count}


async def node_force_approve(state: State) -> dict:
    await _force_approve_best_concept(state["project_id"], _db)
    return {"kill_switch_status": "approved"}


async def _force_approve_best_concept(project_id: str, db) -> None:
    """After 3 exhausted regen rounds, promote the highest-scoring concept regardless of tier."""
    doc = await db["ideation"].find_one({"project_id": str(project_id)}) or {}
    concept_portfolio = doc.get("concept_portfolio", [])
    if not concept_portfolio:
        raise ValueError(f"No concepts in portfolio for force-approval | project_id={project_id}")

    scored = [c for c in concept_portfolio if "kill_score" in c]
    if not scored:
        raise ValueError(f"No scored concepts for force-approval | project_id={project_id}")

    best = max(
        scored,
        key=lambda c: (c.get("kill_score", 0), c.get("format_adherence_score", 0)),
    )

    phase_2_output = doc.get("phase_2_output", {})
    phase_2_output["approved_concepts"] = [best]
    phase_2_output["force_approved"] = True
    phase_2_output["force_approved_reason"] = (
        "3 regeneration rounds exhausted — best available concept promoted"
    )

    await db["ideation"].update_one(
        {"project_id": str(project_id)},
        {"$set": {"phase_2_output": phase_2_output, "updated_at": time.time()}},
        upsert=True,
    )
    logger.warning(
        "force_approve: promoted concept_id=%s kill_score=%.1f after 3 regen rounds | project_id=%s",
        best.get("concept_id"), best.get("kill_score", 0), project_id,
    )


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_by_format_group(state: State) -> str:
    return "phase2_v" if state.get("format_group") == "V" else "phase2_n"


def route_after_kill_switch(state: State) -> str:
    """Routes the graph after concept_kill_switch based on approval status and regen count."""
    if state.get("kill_switch_status") == "approved":
        return "approved"
    elif state.get("regen_count", 0) < 3:
        return "regen"
    else:
        return "force_approve"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_workflow = StateGraph(State)

# Shared setup
_workflow.add_node("creative_constraint_manager",  node_creative_constraint_manager)
_workflow.add_node("constraint_priority_resolver", node_constraint_priority_resolver)
_workflow.add_node("idea_core_preservation",       node_idea_core_preservation)
_workflow.add_node("brand_guideline_alignment",    node_brand_guideline_alignment)
_workflow.add_node("video_type_conditioning",      node_video_type_conditioning)
_workflow.add_node("video_type_selection",         node_video_type_selection)
_workflow.add_node("format_group_detector",        node_format_group_detector)
_workflow.add_node("duration_structuring",         node_duration_structuring)
_workflow.add_node("scene_deconstruction",         node_scene_deconstruction)

# Phase 2-N path
_workflow.add_node("scene_role_enumeration",       node_scene_role_enumeration)
_workflow.add_node("scene_role_selector",          node_scene_role_selector)
_workflow.add_node("temporal_placement_solver",    node_temporal_placement_solver)
_workflow.add_node("scene_integration_plan",       node_scene_integration_plan)
_workflow.add_node("narrative_skeleton_generator", node_narrative_skeleton_generator)
_workflow.add_node("narrative_skeleton_planner",   node_narrative_skeleton_planner)
_workflow.add_node("narrative_archetype_selector", node_narrative_archetype_selector)

# Phase 2-V path
_workflow.add_node("visual_structure_agent",       node_visual_structure_agent)
_workflow.add_node("visual_motif_selector",        node_visual_motif_selector)

# Shared continuation
_workflow.add_node("platform_behavior_optimizer",  node_platform_behavior_optimizer)
_workflow.add_node("pattern_interrupt_generator",  node_pattern_interrupt_generator)
_workflow.add_node("viral_mechanics",              node_viral_mechanics)
_workflow.add_node("mental_model_transformer",     node_mental_model_transformer)
_workflow.add_node("intergalactic_thinking",       node_intergalactic_thinking)
_workflow.add_node("diversity_manifest_generator", node_diversity_manifest_generator)
_workflow.add_node("concept_generator",            node_concept_generator)
_workflow.add_node("phase2_concept_reviewer",      node_phase2_concept_reviewer)
_workflow.add_node("offer_narrative_integrator",   node_offer_narrative_integrator)
_workflow.add_node("concept_categorization",       node_concept_categorization)
_workflow.add_node("interest_filter",              node_interest_filter)
_workflow.add_node("concept_kill_switch",          node_concept_kill_switch)
_workflow.add_node("concept_mutation",             node_concept_mutation)
_workflow.add_node("force_approve",                node_force_approve)

# Shared setup edges
_workflow.add_edge(START,                          "creative_constraint_manager")
_workflow.add_edge("creative_constraint_manager",  "constraint_priority_resolver")
_workflow.add_edge("constraint_priority_resolver", "idea_core_preservation")
_workflow.add_edge("idea_core_preservation",       "brand_guideline_alignment")
_workflow.add_edge("brand_guideline_alignment",    "video_type_conditioning")
_workflow.add_edge("video_type_conditioning",      "video_type_selection")
_workflow.add_edge("video_type_selection",         "format_group_detector")
_workflow.add_edge("format_group_detector",        "duration_structuring")
_workflow.add_edge("duration_structuring",         "scene_deconstruction")

# Format-group fork from scene_deconstruction
_workflow.add_conditional_edges(
    "scene_deconstruction",
    route_by_format_group,
    {
        "phase2_n": "scene_role_enumeration",
        "phase2_v": "visual_structure_agent",
    },
)

# Phase 2-N edges
_workflow.add_edge("scene_role_enumeration",       "scene_role_selector")
_workflow.add_edge("scene_role_selector",          "temporal_placement_solver")
_workflow.add_edge("temporal_placement_solver",    "scene_integration_plan")
_workflow.add_edge("scene_integration_plan",       "narrative_skeleton_generator")
_workflow.add_edge("narrative_skeleton_generator", "narrative_skeleton_planner")
_workflow.add_edge("narrative_skeleton_planner",   "narrative_archetype_selector")
_workflow.add_edge("narrative_archetype_selector", "platform_behavior_optimizer")

# Phase 2-V edges
_workflow.add_edge("visual_structure_agent",       "visual_motif_selector")
_workflow.add_edge("visual_motif_selector",        "platform_behavior_optimizer")

# Shared continuation edges
_workflow.add_edge("platform_behavior_optimizer",  "pattern_interrupt_generator")
_workflow.add_edge("pattern_interrupt_generator",  "viral_mechanics")
_workflow.add_edge("viral_mechanics",              "mental_model_transformer")
_workflow.add_edge("mental_model_transformer",     "intergalactic_thinking")
_workflow.add_edge("intergalactic_thinking",       "diversity_manifest_generator")
_workflow.add_edge("diversity_manifest_generator", "concept_generator")
_workflow.add_edge("concept_generator",            "phase2_concept_reviewer")
_workflow.add_edge("phase2_concept_reviewer",      "offer_narrative_integrator")
_workflow.add_edge("offer_narrative_integrator",   "concept_categorization")
_workflow.add_edge("concept_categorization",       "interest_filter")
_workflow.add_edge("interest_filter",              "concept_kill_switch")

# Regen loop: kill_switch → approved (END) | needs_regen → mutation → interest_filter loop | force_approve (END)
_workflow.add_conditional_edges(
    "concept_kill_switch",
    route_after_kill_switch,
    {"approved": END, "regen": "concept_mutation", "force_approve": "force_approve"},
)
_workflow.add_edge("concept_mutation",             "interest_filter")  # loops back
_workflow.add_edge("force_approve",                END)

graph = _workflow.compile()


# ---------------------------------------------------------------------------
# Public pipeline runner
# ---------------------------------------------------------------------------

async def run_phase_2_pipeline(project_id: str, db) -> dict:
    global _db
    _db = db

    try:
        await _db.ideation.update_one(
            {"project_id": project_id},
            {"$set": {"pipeline_status": "running"}},
            upsert=True,
        )

        pipeline_doc = await _db.pipeline.find_one({"project_id": project_id})
        if not pipeline_doc:
            await _db.pipeline.insert_one({
                "project_id": project_id,
                "phase": 2,
                "created_at": datetime.utcnow(),
                "agent_logs": [],
            })

        logger.info(f"Starting Phase 2 LangGraph pipeline for project_id: {project_id}")

        await graph.ainvoke({
            "project_id": project_id,
            "format_group": "N",      # default — overwritten by format_group_detector node
            "regen_count": 0,         # starts at 0, incremented by concept_mutation node
            "kill_switch_status": "needs_regen",  # default — overwritten by kill_switch node
        })

        await _db.ideation.update_one(
            {"project_id": project_id},
            {"$set": {"pipeline_status": "completed"}},
        )

        logger.info(f"Phase 2 pipeline completed for project_id: {project_id}")
        return {"project_id": project_id, "status": "completed"}

    except Exception as exc:
        import traceback
        logger.error(
            f"Phase 2 pipeline failed for project_id {project_id}: {exc}\n"
            + traceback.format_exc()
        )
        await _db.ideation.update_one(
            {"project_id": project_id},
            {"$set": {"pipeline_status": "failed", "pipeline_error": str(exc)}},
        )
        return {"project_id": project_id, "status": "failed", "error": str(exc)}
