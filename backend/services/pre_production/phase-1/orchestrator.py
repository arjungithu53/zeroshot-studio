import logging
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

# Import the 11 Phase 1 Agents
from brand_adjective import run_brand_adjective_agent
from audience_persona import run_audience_persona_agent
from competitive_landscape import run_competitive_landscape_agent
from central_human_truth import run_central_human_truth_agent
from value_prop_and_offer import run_value_prop_and_offer_agent
from truest_thing import run_truest_thing_agent
from conflict_identification import run_conflict_identification_agent
from insight_validation import run_insight_validation_agent
from truth_conflict_platform import run_truth_conflict_platform_agent
from strategy_models import run_strategy_models_agent
from positioning_alignment import run_positioning_alignment_agent

logger = logging.getLogger("zeroshot.phase1_pipeline")

# 1. Context Dict (State) Schema
class State(TypedDict):
    project_id: str

# 2. Node Wrapper Functions
async def node_brand_adjective(state: State, db: Any) -> dict:
    await run_brand_adjective_agent(state["project_id"], db)
    return {}

async def node_audience_persona(state: State, db: Any) -> dict:
    await run_audience_persona_agent(state["project_id"], db)
    return {}

async def node_competitive_landscape(state: State, db: Any) -> dict:
    await run_competitive_landscape_agent(state["project_id"], db)
    return {}

async def node_central_human_truth(state: State, db: Any) -> dict:
    await run_central_human_truth_agent(state["project_id"], db)
    return {}

async def node_value_prop_and_offer(state: State, db: Any) -> dict:
    await run_value_prop_and_offer_agent(state["project_id"], db)
    return {}

async def node_truest_thing(state: State, db: Any) -> dict:
    await run_truest_thing_agent(state["project_id"], db)
    return {}

async def node_conflict_identification(state: State, db: Any) -> dict:
    await run_conflict_identification_agent(state["project_id"], db)
    return {}

async def node_insight_validation(state: State, db: Any) -> dict:
    await run_insight_validation_agent(state["project_id"], db)
    return {}

async def node_truth_conflict_platform(state: State, db: Any) -> dict:
    await run_truth_conflict_platform_agent(state["project_id"], db)
    return {}

async def node_strategy_models(state: State, db: Any) -> dict:
    await run_strategy_models_agent(state["project_id"], db)
    return {}

async def node_positioning_alignment(state: State, db: Any) -> dict:
    await run_positioning_alignment_agent(state["project_id"], db)
    return {}

# 3. StateGraph Assembly
def build_phase1_graph(db: Any):
    workflow = StateGraph(State)

    async def _brand_adjective(s): return await node_brand_adjective(s, db)
    async def _audience_persona(s): return await node_audience_persona(s, db)
    async def _competitive_landscape(s): return await node_competitive_landscape(s, db)
    async def _central_human_truth(s): return await node_central_human_truth(s, db)
    async def _value_prop_and_offer(s): return await node_value_prop_and_offer(s, db)
    async def _truest_thing(s): return await node_truest_thing(s, db)
    async def _conflict_identification(s): return await node_conflict_identification(s, db)
    async def _insight_validation(s): return await node_insight_validation(s, db)
    async def _truth_conflict_platform(s): return await node_truth_conflict_platform(s, db)
    async def _strategy_models(s): return await node_strategy_models(s, db)
    async def _positioning_alignment(s): return await node_positioning_alignment(s, db)

    # Add Nodes
    workflow.add_node("brand_adjective", _brand_adjective)
    workflow.add_node("audience_persona", _audience_persona)
    workflow.add_node("competitive_landscape", _competitive_landscape)
    workflow.add_node("central_human_truth", _central_human_truth)
    workflow.add_node("value_prop_and_offer", _value_prop_and_offer)
    workflow.add_node("truest_thing", _truest_thing)
    workflow.add_node("conflict_identification", _conflict_identification)
    workflow.add_node("insight_validation", _insight_validation)
    workflow.add_node("truth_conflict_platform", _truth_conflict_platform)
    workflow.add_node("strategy_models", _strategy_models)
    workflow.add_node("positioning_alignment", _positioning_alignment)

    # Set Entry Point
    workflow.set_entry_point("brand_adjective")

    # Add Edges (Linear sequence)
    workflow.add_edge("brand_adjective", "audience_persona")
    workflow.add_edge("audience_persona", "competitive_landscape")
    workflow.add_edge("competitive_landscape", "central_human_truth")
    workflow.add_edge("central_human_truth", "value_prop_and_offer")
    workflow.add_edge("value_prop_and_offer", "truest_thing")
    workflow.add_edge("truest_thing", "insight_validation")
    workflow.add_edge("insight_validation", "conflict_identification")
    workflow.add_edge("conflict_identification", "truth_conflict_platform")
    workflow.add_edge("truth_conflict_platform", "strategy_models")
    workflow.add_edge("strategy_models", "positioning_alignment")
    workflow.add_edge("positioning_alignment", END)

    return workflow.compile()

# 4. Pipeline Runner Function
async def run_phase_1_pipeline(project_id: str, db: Any) -> dict:
    """
    Kicks off the full Phase 1 Agent pipeline using LangGraph orchestration.
    """
    try:
        logger.info(f"Starting Phase 1 pipeline for project_id: {project_id}")
        graph = build_phase1_graph(db)
        # Execute the pipeline
        await graph.ainvoke({"project_id": project_id})
        logger.info(f"Phase 1 pipeline completed successfully for project_id: {project_id}")
        return {
            "project_id": project_id,
            "status": "success",
            "message": "Phase 1 pipeline executed completely."
        }
    except Exception as e:
        logger.error(f"Error in Phase 1 pipeline for project_id: {project_id}: {str(e)}")
        # You could add further logic here to update a strategy or pipeline execution document with the failure.
        return {
            "project_id": project_id,
            "status": "failed",
            "error": str(e)
        }
