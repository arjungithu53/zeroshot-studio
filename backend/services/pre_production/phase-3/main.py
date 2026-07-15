import logging
import os
import sys
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "beat_to_timeline_mapper"))
sys.path.insert(0, os.path.join(_HERE, "offer_integration_planner"))
sys.path.insert(0, os.path.join(_HERE, "visual_sequencing_agent"))
sys.path.insert(0, os.path.join(_HERE, "av_separation_agent"))
sys.path.insert(0, os.path.join(_HERE, "voiceover_writer_agent"))
sys.path.insert(0, os.path.join(_HERE, "dialogue_agent"))
sys.path.insert(0, os.path.join(_HERE, "audio_design_agent"))
sys.path.insert(0, os.path.join(_HERE, "rhythm_pacing_regulator"))
sys.path.insert(0, os.path.join(_HERE, "loop_optimization_agent"))
sys.path.insert(0, os.path.join(_HERE, "constraint_compliance_qa_agent"))
sys.path.insert(0, os.path.join(_HERE, "shot_level_script_formatter"))
sys.path.insert(0, os.path.join(_HERE, 'final_shotlist_agent'))
# sys.path.insert(0, os.path.join(_HERE, 'reviewer_3'))
# sys.path.insert(0, os.path.join(_HERE, 'reviewer_5'))
# sys.path.insert(0, os.path.join(_HERE, 'reviewer_6'))
# sys.path.insert(0, os.path.join(_HERE, 'reviewer_7'))

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

logger = logging.getLogger("zeroshot.phase3.router")

phase3_router = APIRouter()


class BeatToTimelineMapperRequest(BaseModel):
    project_id: str


class BeatToTimelineMapperResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class OfferIntegrationPlannerRequest(BaseModel):
    project_id: str


class OfferIntegrationPlannerResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class VisualSequencingRequest(BaseModel):
    project_id: str


class VisualSequencingResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class AVSeparationRequest(BaseModel):
    project_id: str


class AVSeparationResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class VoiceoverWriterRequest(BaseModel):
    project_id: str


class VoiceoverWriterResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class DialogueAgentRequest(BaseModel):
    project_id: str


class DialogueAgentResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class AudioDesignRequest(BaseModel):
    project_id: str


class AudioDesignResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class RhythmPacingRegulatorRequest(BaseModel):
    project_id: str

class ConstraintComplianceQARequest(BaseModel):
    project_id: str

class ConstraintComplianceQAResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class RhythmPacingRegulatorResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class LoopOptimizationRequest(BaseModel):
    project_id: str


class LoopOptimizationResponse(BaseModel):
    message: str
    data: Dict[str, Any]

class ShotLevelScriptFormatterRequest(BaseModel):
    project_id: str

class ShotLevelScriptFormatterResponse(BaseModel):
    message: str
    data: Dict[str, Any]

class FinalShotAgentRequest(BaseModel):
    project_id: str

class FinalShotAgentResponse(BaseModel):
    message: str
    data:    Dict[str, Any]

class MultiAgentPipelineRequest(BaseModel):
    project_id: str

class MultiAgentPipelineResponse(BaseModel):
    message: str
    data: Dict[str, Any]



@phase3_router.post(
    "/run-multi-agent-pipeline",
    response_model=MultiAgentPipelineResponse,
    summary="Run Phase 3 Pipeline with LangGraph",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def run_multi_agent_pipeline_endpoint(body: MultiAgentPipelineRequest) -> MultiAgentPipelineResponse:
    """
    Triggers the entire Phase 3 LangGraph orchestrator pipeline.
    Runs all 11 agents in sequence, properly routing QA gate logic.
    """
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("phase3_orchestrator", os.path.join(_HERE, "orchestrator.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    run_phase_3_pipeline = _mod.run_phase_3_pipeline
    from main import _db
    
    try:
        result = await run_phase_3_pipeline(body.project_id, _db)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        return MultiAgentPipelineResponse(
            message="Phase 3 Pipeline execution completed.",
            data=result
        )
    except Exception as exc:
        logger.error(f"Error in Phase 3 Pipeline endpoint: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to execute pipeline: {str(exc)}"
        )

@phase3_router.post(
    "/beat-to-timeline-mapper",
    response_model=BeatToTimelineMapperResponse,
    summary="Run Agent 1: beat_to_timeline_mapper",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def beat_to_timeline_mapper_endpoint(body: BeatToTimelineMapperRequest) -> BeatToTimelineMapperResponse:
    """
    Triggers Agent 1 (beat_to_timeline_mapper).
    Converts approved story beats and narrative budget into an exact master timeline.
    """
    from main import _db

    try:
        result = await run_beat_to_timeline_mapper_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 1 beat-to-timeline mapping completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "master_timeline": result.master_timeline.model_dump(),
            "reasoning": result.reasoning,
            "beat_expansion_log": [item.model_dump() for item in result.beat_expansion_log],
            "rationale": result.rationale,
            "timing_decisions": result.timing_decisions,
            "status": result.status,
        },
    }


@phase3_router.post(
    "/offer-integration-planner",
    response_model=OfferIntegrationPlannerResponse,
    summary="Run Agent 2: offer_integration_planner",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def offer_integration_planner_endpoint(body: OfferIntegrationPlannerRequest) -> OfferIntegrationPlannerResponse:
    """
    Triggers Agent 2 (offer_integration_planner).
    Resolves archetype-CTA tension and outputs binding offer constraints for downstream writers.
    """
    from main import _db

    try:
        result = await run_offer_integration_planner_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 2 offer integration planning completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "offer_constraints": result.offer_constraints.model_dump(),
            "reasoning": result.reasoning,
            "status": result.status,
        },
    }


@phase3_router.post(
    "/visual-sequencing",
    response_model=VisualSequencingResponse,
    summary="Run Agent 3: visual_sequencing_agent",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def visual_sequencing_endpoint(body: VisualSequencingRequest) -> VisualSequencingResponse:
    """
    Triggers Agent 3 (visual_sequencing_agent).
    Builds a production-actionable shot list from master timeline, scene intelligence, and offer constraints.
    """
    from main import _db

    try:
        result = await run_visual_sequencing_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 3 visual sequencing completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "shot_list": [shot.model_dump() for shot in result.shot_list],
            "reasoning": result.reasoning,
            "key_frame_rationale": result.key_frame_rationale,
            "mobile_compliance_check": result.mobile_compliance_check,
            "status": result.status,
        },
    }


@phase3_router.post(
    "/av-separation",
    response_model=AVSeparationResponse,
    summary="Run Agent 4: av_separation_agent",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def av_separation_endpoint(body: AVSeparationRequest) -> AVSeparationResponse:
    """
    Triggers Agent 4 (av_separation_agent).
    Maps each time window to determine which piece of information belongs in the visual channel
    and which in the VO channel enforcing the 1+1=3 principle.
    """
    from main import _db

    try:
        result = await run_av_separation_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 4 AV separation completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "av_channel_map": [decision.model_dump() for decision in result.av_channel_map],
            "reasoning": result.reasoning,
            "status": result.status,
        },
    }


@phase3_router.post(
    "/voiceover-writer",
    response_model=VoiceoverWriterResponse,
    summary="Run Agent 5: voiceover_writer_agent",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def voiceover_writer_endpoint(body: VoiceoverWriterRequest) -> VoiceoverWriterResponse:
    """
    Triggers Agent 5 (voiceover_writer_agent).
    Writes the exact spoken words for every VO window in the av_channel_map.
    """
    from main import _db

    try:
        result = await run_voiceover_writer_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 5 voiceover writing completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "vo_script": [line.model_dump() for line in result.vo_script],
            "reasoning": result.reasoning,
            "straight_version_log": result.straight_version_log,
            "lateral_craft_log": result.lateral_craft_log,
            "status": result.status,
        },
    }


@phase3_router.post(
    "/dialogue",
    response_model=DialogueAgentResponse,
    summary="Run Agent 6: dialogue_agent",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def dialogue_endpoint(body: DialogueAgentRequest) -> DialogueAgentResponse:
    """
    Triggers Agent 6 (dialogue_agent).
    Analyzes concept story beats to determine if characters speak directly to each other and scripts natural, authentic dialogue lines avoiding marketing language.
    """
    from main import _db

    try:
        result = await run_dialogue_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 6 dialogue completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "dialogue_lines": [line.model_dump() for line in result.dialogue_lines] if result.dialogue_lines else None,
            "reasoning": result.reasoning,
            "naturalness_review": result.naturalness_review,
            "fragment_analysis": result.fragment_analysis,
            "status": getattr(result, 'status', 'completed'),
        },
    }


@phase3_router.post(
    "/audio-design",
    response_model=AudioDesignResponse,
    summary="Run Agent 7: audio_design_agent",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def audio_design_endpoint(body: AudioDesignRequest) -> AudioDesignResponse:
    """
    Triggers Agent 7 (audio_design_agent).
    Designs non-verbal audio aspects: ambient texture, sound effects, music mood, and silence.
    Ensures sound-off redundancy and preventing narrative collision.
    """
    from main import _db

    try:
        result = await run_audio_design_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 7 audio design completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "windows": [window.model_dump() for window in result.windows] if result.windows else None,
            "music_mood_curve": result.music_mood_curve,
            "sound_off_compliant": result.sound_off_compliant,
            "status": getattr(result, 'status', 'completed'),
        },
    }

@phase3_router.post(
    "/rhythm-pacing-regulator",
    response_model=RhythmPacingRegulatorResponse,
    summary="Run Agent 8: rhythm_pacing_regulator",
    tags=["Agents - Phase 3"],
)
async def execute_rhythm_pacing_regulator_agent(body: RhythmPacingRegulatorRequest):
    """
    Agent 8: Rhythm & Pacing Regulator
    Operates as holistic pacing layer evaluating cut density, VO density, micro-pauses & energy curve.
    """
    from main import _db
    
    try:
        result = await run_rhythm_pacing_regulator_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(f"Validation error | project_id={body.project_id} error={exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Agent runtime error | project_id={body.project_id} error={exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 8 rhythm and pacing regulation completed with status '{result.get('status', 'completed')}' for project '{body.project_id}'",
        "data": result,
    }


@phase3_router.post(
    "/loop-optimization",
    response_model=LoopOptimizationResponse,
    summary="Run Agent 9: loop_optimization_agent",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def loop_optimization_endpoint(body: LoopOptimizationRequest) -> LoopOptimizationResponse:
    """
    Triggers Agent 9 (loop_optimization_agent).
    Evaluates loop continuity across visual, tonal, audio, and curiosity dimensions.
    Returns revision requests to upstream agents if issues are detected.
    """
    from main import _db

    try:
        result = await run_loop_optimization_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(f"Validation error | project_id={body.project_id} error={exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Agent runtime error | project_id={body.project_id} error={exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    data_payload = {
        "visual_continuity_pass": getattr(result, 'visual_continuity_pass', False),
        "tonal_continuity_pass": getattr(result, 'tonal_continuity_pass', False),
        "audio_continuity_pass": getattr(result, 'audio_continuity_pass', False),
        "curiosity_loop_pass": getattr(result, 'curiosity_loop_pass', False),
        "revision_requests": [req.model_dump() for req in getattr(result, 'revision_requests', [])] if getattr(result, 'revision_requests', None) else None,
        "status": getattr(result, 'status', 'completed'),
    }

    return {
        "message": f"Agent 9 loop optimization completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": data_payload,
    }

@phase3_router.post(
    "/constraint-compliance-qa",
    response_model=ConstraintComplianceQAResponse,
    summary="Run Agent 10: constraint_compliance_qa_agent",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def constraint_compliance_qa_endpoint(body: ConstraintComplianceQARequest) -> ConstraintComplianceQAResponse:
    """
    Triggers Agent 10 (constraint_compliance_qa_agent).
    Runs the fully assembled script against every constraint layer from Phase 2.
    """
    from main import _db

    try:
        result = await run_constraint_compliance_qa_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 10 constraint QA completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "qa_result": result.model_dump(by_alias=True),
        },
    }


@phase3_router.post(
    "/shot-level-script-formatter",
    response_model=ShotLevelScriptFormatterResponse,
    summary="Run Agent 11: shot_level_script_formatter",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def shot_level_script_formatter_endpoint(body: ShotLevelScriptFormatterRequest) -> ShotLevelScriptFormatterResponse:
    """
    Triggers Agent 11 (shot_level_script_formatter).
    Serializes QA-cleared script into a valid Fountain format and creates density map.
    """
    from main import _db

    try:
        result = await run_shot_level_script_formatter(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 11 shot level script formatting completed with status '{getattr(result, 'status', 'completed')}' for project '{body.project_id}'",
        "data": {
            "fountain_raw": getattr(result, 'fountain_raw', ''),
            "channel_density_map": getattr(result, 'channel_density_map', ''),
            "constraint_compliance_summary": getattr(result, 'constraint_compliance_summary', ''),
            "reasoning": getattr(result, 'reasoning', ''),
            "fountain_s3_url": getattr(result, 'fountain_s3_url', ''),
            "status": getattr(result, 'status', 'completed'),
        },
    }


@phase3_router.post(
    "/final-shotlist-agent",
    response_model=FinalShotAgentResponse,
    summary="Run Agent 12: final_shotlist_agent",
    tags=["Agents - Phase 3"],
    status_code=200,
)
async def final_shot_agent_endpoint(
    body: FinalShotAgentRequest
) -> FinalShotAgentResponse:
    """
    Triggers Agent 12 (final_shotlist_agent).
    Reads the formatted script from Agent 11 and produces a
    production shot list CSV with one row per camera setup,
    including a product_present column.
    """
    from main import _db
    
    try:
        result = await run_final_shot_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s",
            body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s",
            body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return FinalShotAgentResponse(
        message=f"Agent 12 final shot list completed with status "
                f"'{getattr(result, 'status', 'completed')}' "
                f"for project '{body.project_id}'",
        data={
            "shotlist_s3_url": result.shotlist_s3_url,
            "shots_count":     len(result.shots),
            "reasoning":       result.reasoning,
            "status":          result.status,
        }
    )