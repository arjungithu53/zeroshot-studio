import logging
import os
from datetime import datetime
from typing import TypedDict, Any
from bson import ObjectId

_db = None

from langgraph.graph import StateGraph, START, END

from revision_utils import REVISION_SUFFIX, build_revision_prompt_prefix

# Import the agent runner functions
from beat_to_timeline_mapper import run_beat_to_timeline_mapper_agent
from offer_integration_planner import run_offer_integration_planner_agent
from visual_sequencing_agent import run_visual_sequencing_agent
from av_separation_agent import run_av_separation_agent
from voiceover_writer_agent import run_voiceover_writer_agent
from dialogue_agent import run_dialogue_agent
from audio_design_agent import run_audio_design_agent
from rhythm_pacing_regulator import run_rhythm_pacing_regulator_agent
from loop_optimization_agent import run_loop_optimization_agent
from constraint_compliance_qa_agent import run_constraint_compliance_qa_agent
from shot_level_script_formatter import run_shot_level_script_formatter
from final_shotlist_agent import run_final_shot_agent

from reviewer_3.main import run_reviewer_3
from reviewer_5.main import run_reviewer_5
from reviewer_6.main import run_reviewer_6
from reviewer_7.main import run_reviewer_7

logger = logging.getLogger("zeroshot.phase3.orchestrator")

class State(TypedDict):
    project_id:   str
    pipeline_id:  str
    qa_status:    str
    qa_result:    dict | None

# Node Wrapper Functions
async def node_beat_to_timeline_mapper(state: State) -> dict:
    await run_beat_to_timeline_mapper_agent(state['project_id'], _db)
    return {}

async def node_offer_integration_planner(state: State) -> dict:
    await run_offer_integration_planner_agent(state['project_id'], _db)
    return {}

async def node_visual_sequencing_agent(state: State) -> dict:
    await run_visual_sequencing_agent(state['project_id'], _db)
    return {}

async def node_reviewer_3(state: State) -> dict:
    """Reviewer node for visual_sequencing_agent output."""
    result = await run_reviewer_3(state['project_id'], _db)
    return result

async def node_av_separation_agent(state: State) -> dict:
    await run_av_separation_agent(state['project_id'], _db)
    return {}

async def node_voiceover_writer_agent(state: State) -> dict:
    await run_voiceover_writer_agent(state['project_id'], _db)
    return {}

async def node_reviewer_5(state: State) -> dict:
    """Reviewer node for voiceover_writer_agent output."""
    result = await run_reviewer_5(state['project_id'], _db)
    return result

async def node_dialogue_agent(state: State) -> dict:
    await run_dialogue_agent(state['project_id'], _db)
    return {}

async def node_reviewer_6(state: State) -> dict:
    """Reviewer node for dialogue_agent output."""
    result = await run_reviewer_6(state['project_id'], _db)
    return result

async def node_audio_design_agent(state: State) -> dict:
    await run_audio_design_agent(state['project_id'], _db)
    return {}

async def node_reviewer_7(state: State) -> dict:
    """Reviewer node for audio_design_agent output."""
    result = await run_reviewer_7(state['project_id'], _db)
    return result

async def node_rhythm_pacing_regulator(state: State) -> dict:
    await run_rhythm_pacing_regulator_agent(state['project_id'], _db)
    return {}

_LOOP_REVISION_AGENT_MAP = {
    "visual_sequencing_agent": (run_visual_sequencing_agent, 3),
    "voiceover_writer_agent":  (run_voiceover_writer_agent,  5),
    "audio_design_agent":      (run_audio_design_agent,      7),
}

async def node_loop_optimization_agent(state: State) -> dict:
    result = await run_loop_optimization_agent(state['project_id'], _db)

    if result.revision_requests:
        by_agent = {}
        for req in result.revision_requests:
            by_agent.setdefault(req.target_agent, []).append(req)

        for target_name, requests in by_agent.items():
            entry = _LOOP_REVISION_AGENT_MAP.get(target_name)
            if entry is None:
                logger.warning(f"Loop optimization: unknown revision target '{target_name}', skipping.")
                continue
            agent_fn, agent_num = entry
            revision_prefix = build_revision_prompt_prefix(
                cycle=1,
                target_agent=agent_num,
                instructions=[
                    {'dimension': r.dimension, 'instruction': r.instruction}
                    for r in requests
                ]
            )
            await agent_fn(state['project_id'], _db, revision_prefix=revision_prefix)

    return {}

async def node_constraint_compliance_qa_agent(state: State) -> dict:
    result = await run_constraint_compliance_qa_agent(
        state['project_id'], _db
    )
    qa_result = result.model_dump() if hasattr(result, 'model_dump') \
                else dict(result)
    return {
        'qa_status': qa_result.get('status', ''),
        'qa_result': qa_result
    }

async def node_shot_level_script_formatter(state: State) -> dict:
    await run_shot_level_script_formatter(state['project_id'], _db)
    return {}

async def node_final_shot_agent(state: State) -> dict:
    """Node wrapper for final_shotlist_agent (Agent 12)."""
    await run_final_shot_agent(state['project_id'], _db)
    return {}

# Build StateGraph
workflow = StateGraph(State)

workflow.add_node('beat_to_timeline_mapper',        node_beat_to_timeline_mapper)
workflow.add_node('offer_integration_planner',      node_offer_integration_planner)
workflow.add_node('visual_sequencing_agent',        node_visual_sequencing_agent)
workflow.add_node('reviewer_3',                     node_reviewer_3)
workflow.add_node('av_separation_agent',            node_av_separation_agent)
workflow.add_node('voiceover_writer_agent',         node_voiceover_writer_agent)
workflow.add_node('reviewer_5',                     node_reviewer_5)
workflow.add_node('dialogue_agent',                 node_dialogue_agent)
workflow.add_node('reviewer_6',                     node_reviewer_6)
workflow.add_node('audio_design_agent',             node_audio_design_agent)
workflow.add_node('reviewer_7',                     node_reviewer_7)
workflow.add_node('rhythm_pacing_regulator',        node_rhythm_pacing_regulator)
workflow.add_node('loop_optimization_agent',        node_loop_optimization_agent)
workflow.add_node('constraint_compliance_qa_agent', node_constraint_compliance_qa_agent)
workflow.add_node('shot_level_script_formatter',    node_shot_level_script_formatter)
workflow.add_node('final_shot_agent',               node_final_shot_agent)

workflow.add_edge(START,                            'beat_to_timeline_mapper')
workflow.add_edge('beat_to_timeline_mapper',        'offer_integration_planner')
workflow.add_edge('offer_integration_planner',      'visual_sequencing_agent')
workflow.add_edge('visual_sequencing_agent',        'reviewer_3')
workflow.add_edge('reviewer_3',                     'av_separation_agent')
workflow.add_edge('av_separation_agent',            'voiceover_writer_agent')
workflow.add_edge('voiceover_writer_agent',         'reviewer_5')
workflow.add_edge('reviewer_5',                     'dialogue_agent')
workflow.add_edge('dialogue_agent',                 'reviewer_6')
workflow.add_edge('reviewer_6',                     'audio_design_agent')
workflow.add_edge('audio_design_agent',             'reviewer_7')
workflow.add_edge('reviewer_7',                     'rhythm_pacing_regulator')
workflow.add_edge('rhythm_pacing_regulator',        'loop_optimization_agent')
workflow.add_edge('loop_optimization_agent',        'constraint_compliance_qa_agent')
workflow.add_edge('constraint_compliance_qa_agent', 'shot_level_script_formatter')
workflow.add_edge('shot_level_script_formatter',    'final_shot_agent')
workflow.add_edge('final_shot_agent',               END)

graph = workflow.compile()

# Set up pipeline runner
async def run_phase_3_pipeline(project_id: str, db) -> dict:
    global _db
    _db = db

    try:
        await _db.script.update_one(
            {'project_id': project_id},
            {'$set': {'status': 'running'}},
            upsert=True
        )

        pipeline_doc = await _db.pipeline.find_one({'project_id': project_id})
        if not pipeline_doc:
            insert_result = await _db.pipeline.insert_one({
                'project_id': project_id,
                'phase': 3,
                'created_at': datetime.utcnow(),
                'agent_logs': []
            })
            pipeline_id = str(insert_result.inserted_id)
        else:
            pipeline_id = str(pipeline_doc['_id'])

        logger.info(f"Starting Phase 3 LangGraph Pipeline for project_id: {project_id}")

        result_state = await graph.ainvoke({
            'project_id':  project_id,
            'pipeline_id': pipeline_id,
            'qa_status':   '',
            'qa_result':   None,
        })

        # Agent 10 logs qa_result but pipeline always continues to Agent 11.
        # Script status reflects whether Agent 11 completed, not QA pass/fail.
        qa = result_state.get('qa_result') or {}
        qa_status = qa.get('status', 'unknown')
        await _db.script.update_one(
            {'project_id': project_id},
            {'$set': {'status': 'completed', 'qa_status': qa_status}}
        )
        return {'project_id': project_id, 'status': 'completed', 'qa_status': qa_status}

    except Exception as e:
        import traceback
        if "503" in str(e) or "429" in str(e):
            logger.error(f"Gemini API limit reached. Pipeline halted gracefully. Please try again.")
        else:
            logger.error(f"Error in Phase 3 Pipeline for {project_id}: {str(e)}\n{traceback.format_exc()}")
        await _db.script.update_one(
            {'project_id': project_id},
            {'$set': {'status': 'failed', 'error_log': str(e)}}
        )
        return {'project_id': project_id, 'status': 'error', 'message': str(e)}
