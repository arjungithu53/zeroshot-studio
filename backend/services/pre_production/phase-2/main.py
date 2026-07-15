import os
import sys

# Define phase 2 paths for agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "creative_constraint_manager"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "constraint_priority_resolver"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "idea_core_preservation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "brand_guideline_alignment"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_type_selection_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "duration_structuring_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_deconstruction_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_role_enumeration_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_role_selector_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "temporal_placement_solver_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_integration_plan"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "narrative_skeleton_generator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "narrative_skeleton_planner"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "narrative_archetype_selector"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "platform_behavior_optimizer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pattern_interrupt_generator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "viral_mechanics_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mental_model_transformer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "intergalactic_thinking_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "concept_generator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "offer_narrative_integrator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "concept_categorization_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_structure_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_motif_selector"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "diversity_manifest_generator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_concept_reviewer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "concept_mutation_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "interest_filter_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "concept_kill_switch"))

from fastapi import APIRouter, Request, HTTPException
import logging

logger = logging.getLogger("zeroshot.phase2.router")

from pydantic import BaseModel
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
from platform_behavior_optimizer import run_platform_behavior_optimizer_agent
from pattern_interrupt_generator import run_pattern_interrupt_generator_agent
from viral_mechanics_agent import run_viral_mechanics_agent
from mental_model_transformer import run_mental_model_transformer
from intergalactic_thinking_agent import run_intergalactic_thinking_agent
from concept_generator import run_concept_generator_agent
from offer_narrative_integrator import run_offer_narrative_integrator_agent
from concept_categorization_agent import run_concept_categorization_agent
from visual_structure_agent import run_visual_structure_agent
from visual_motif_selector import run_visual_motif_selector
from diversity_manifest_generator import run_diversity_manifest_generator
from phase2_concept_reviewer import run_phase2_concept_reviewer
from interest_filter_agent import run_interest_filter_agent
from concept_kill_switch import run_concept_kill_switch_agent
from orchestrator import run_phase_2_pipeline

# ---------------------------------------------------------------------------
# Phase 2 FastAPI Router
# ---------------------------------------------------------------------------
phase2_router = APIRouter()

# Request/Response models for routes
class CreativeConstraintManagerRequest(BaseModel):
    project_id: str

class CreativeConstraintManagerResponse(BaseModel):
    message: str
    data: dict

class ConstraintPriorityResolverRequest(BaseModel):
    project_id: str

class ConstraintPriorityResolverResponse(BaseModel):
    message: str
    data: dict

class IdeaCorePreservationRequest(BaseModel):
    project_id: str

class IdeaCorePreservationResponse(BaseModel):
    message: str
    data: dict

class BrandGuidelineAlignmentRequest(BaseModel):
    project_id: str

class BrandGuidelineAlignmentResponse(BaseModel):
    message: str
    data: dict

class VideoTypeConditioningRequest(BaseModel):
    project_id: str

class VideoTypeConditioningResponse(BaseModel):
    message: str
    data: dict

class VideoTypeSelectionRequest(BaseModel):
    project_id: str

class VideoTypeSelectionResponse(BaseModel):
    message: str
    data: dict

class DurationStructuringRequest(BaseModel):
    project_id: str

class DurationStructuringResponse(BaseModel):
    message: str
    data: dict

class ViralMechanicsRequest(BaseModel):
    project_id: str

class ViralMechanicsResponse(BaseModel):
    message: str
    data: dict

# Include other agents' route defs as needed...

class SceneDeconstructionRequest(BaseModel):
    project_id: str

class SceneDeconstructionResponse(BaseModel):
    message: str
    data: dict

class SceneRoleEnumerationRequest(BaseModel):
    project_id: str

class SceneRoleEnumerationAPIResponse(BaseModel):
    message: str
    data: dict

class TemporalPlacementSolverRequest(BaseModel):
    project_id: str

class TemporalPlacementSolverAPIResponse(BaseModel):
    message: str
    data: dict

class SceneIntegrationPlanRequest(BaseModel):
    project_id: str

class SceneIntegrationPlanResponse(BaseModel):
    message: str
    data: dict

class NarrativeSkeletonGeneratorRequest(BaseModel):
    project_id: str

class NarrativeSkeletonGeneratorResponse(BaseModel):
    message: str
    data: dict

class SceneRoleSelectorRequest(BaseModel):
    project_id: str

class SceneRoleSelectorAPIResponse(BaseModel):
    message: str
    data: dict

class TemporalPlacementSolverRequest(BaseModel):
    project_id: str

class TemporalPlacementSolverAPIResponse(BaseModel):
    message: str
    data: dict

class NarrativeSkeletonPlannerRequest(BaseModel):
    project_id: str

class NarrativeSkeletonPlannerResponse(BaseModel):
    message: str
    data: dict

class NarrativeArchetypeSelectorRequest(BaseModel):
    project_id: str

class NarrativeArchetypeSelectorResponse(BaseModel):
    message: str
    data: dict

class PlatformBehaviorOptimizerRequest(BaseModel):
    project_id: str

class PlatformBehaviorOptimizerResponse(BaseModel):
    message: str
    data: dict

class PatternInterruptGeneratorRequest(BaseModel):
    project_id: str

class PatternInterruptGeneratorResponse(BaseModel):
    message: str
    data: dict

class MentalModelTransformerRequest(BaseModel):
    project_id: str

class MentalModelTransformerResponse(BaseModel):
    message: str
    data: dict

class IntergalacticThinkingRequest(BaseModel):
    project_id: str

class IntergalacticThinkingResponse(BaseModel):
    message: str
    data: dict

class ConceptGeneratorRequest(BaseModel):
    project_id: str

class ConceptGeneratorResponse(BaseModel):
    message: str
    data: dict

class OfferNarrativeIntegratorRequest(BaseModel):
    project_id: str

class OfferNarrativeIntegratorResponse(BaseModel):
    message: str
    data: dict

class ConceptCategorizationRequest(BaseModel):
    project_id: str

class ConceptCategorizationResponse(BaseModel):
    message: str
    data: dict

class ConceptDiversityControllerRequest(BaseModel):
    project_id: str

class ConceptDiversityControllerResponse(BaseModel):
    message: str
    data: dict

class InterestFilterAgentRequest(BaseModel):
    project_id: str

class InterestFilterAgentResponse(BaseModel):
    message: str
    data: dict

class ConceptKillSwitchRequest(BaseModel):
    project_id: str

class ConceptKillSwitchResponse(BaseModel):
    message: str
    data: dict

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@phase2_router.post(
    "/creative-constraint-manager",
    response_model=CreativeConstraintManagerResponse,
    summary="Run Agent 1: Creative Constraint Manager",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def creative_constraint_manager_endpoint(body: CreativeConstraintManagerRequest) -> CreativeConstraintManagerResponse:
    """
    Triggers the Entry Orchestration Agent (Agent 1).
    Ingests strategy and projects, outputs a constraint graph.
    """
    # Import the global _db initialized in Phase 1's lifespan
    from main import _db

    try:
        result = await run_creative_constraint_manager_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 1 constraint graph created successfully for project '{body.project_id}'",
        # Send raw data mapping
        "data": {
            "hard_constraints": result.hard_constraints,
            "feasibility_envelope": result.feasibility_envelope,
            "idea_classification_detail": result.idea_classification_detail
        }
    }

@phase2_router.post(
    "/constraint-priority-resolver",
    response_model=ConstraintPriorityResolverResponse,
    summary="Run Agent 2: Constraint Priority Resolver",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def constraint_priority_resolver_endpoint(body: ConstraintPriorityResolverRequest) -> ConstraintPriorityResolverResponse:
    """
    Triggers the Constraint Priority Resolver Agent (Agent 2).
    Resolves constraint conflicts and emits structured directives.
    """
    from main import _db

    try:
        result = await run_constraint_priority_resolver_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 2 constraint priority directives created successfully for project '{body.project_id}'",
        "data": {
            "resolved_conflicts": [log.model_dump() for log in result.priority_directives.resolved_conflicts],
            "operating_mode": result.priority_directives.operating_mode,
            "operating_mode_rationale": result.operating_mode_rationale
        }
    }


@phase2_router.post(
    "/idea-core-preservation",
    response_model=IdeaCorePreservationResponse,
    summary="Run Agent 3: Idea Core Preservation",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def idea_core_preservation_endpoint(body: IdeaCorePreservationRequest) -> IdeaCorePreservationResponse:
    """
    Triggers the Idea Core Preservation Agent (Agent 3).
    Defines structural integrity rules when a brand idea is provided.
    """
    from main import _db

    try:
        result = await run_idea_core_preservation_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 3 idea core rules processed for project '{body.project_id}'",
        "data": result.model_dump()
    }


@phase2_router.post(
    "/brand-guideline-alignment",
    response_model=BrandGuidelineAlignmentResponse,
    summary="Run Agent 4: Brand Guideline Alignment",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def brand_guideline_alignment_endpoint(body: BrandGuidelineAlignmentRequest) -> BrandGuidelineAlignmentResponse:
    """
    Triggers the Brand Guideline Alignment Agent (Agent 4).
    Translates brand mandatories and adjectives into concrete tonal guardrails and cultural modulation rules.
    """
    from main import _db

    try:
        result = await run_brand_guideline_alignment_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 4 brand guideline alignment processed for project {body.project_id!r}",
        "data": result
    }

@phase2_router.post(
    "/video-type-conditioning",
    response_model=VideoTypeConditioningResponse,
    summary="Run Agent 5a: Video Type Conditioning",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def video_type_conditioning_endpoint(body: VideoTypeConditioningRequest) -> VideoTypeConditioningResponse:
    """
    Triggers the Video Type Conditioning Agent (Agent 5a).
    When video type is confirmed, conditions the creative search space for that format.
    """
    from main import _db

    try:
        result = await run_video_type_conditioning_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 5a video type conditioning processed for project {body.project_id!r}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }

@phase2_router.post(
    "/video-type-selection",
    response_model=VideoTypeSelectionResponse,
    summary="Run Agent 5b: Video Type Selection",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def video_type_selection_endpoint(body: VideoTypeSelectionRequest) -> VideoTypeSelectionResponse:
    """
    Triggers the Video Type Selection Agent (Agent 5b).
    When video type is not specified, evaluates viable formats and selects the optimal format, then conditions the search space.
    """
    from main import _db

    try:
        result = await run_video_type_selection_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Agent 5b video type selection processed for project {body.project_id!r}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }
@phase2_router.post(
    "/duration-structuring",
    response_model=DurationStructuringResponse,
    summary="Run Agent 6: Duration Structuring Agent",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def duration_structuring_endpoint(body: DurationStructuringRequest) -> DurationStructuringResponse:
    from main import _db

    try:
        result = await run_duration_structuring_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Duration Structuring (Agent 6) executed. Project: {body.project_id}. Status: {result.status}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }

@phase2_router.post(
    "/run-scene-deconstruction",
    response_model=SceneDeconstructionResponse,
    summary="Run Agent 7: Scene Deconstruction Agent",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def scene_deconstruction_endpoint(body: SceneDeconstructionRequest) -> SceneDeconstructionResponse:
    from main import _db

    try:
        result = await run_scene_deconstruction_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Scene Deconstruction (Agent 7) executed. Project: {body.project_id}. Status: {getattr(result, 'status', 'completed')}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }

@phase2_router.post(
    "/run-scene-role-enumeration",
    response_model=SceneRoleEnumerationAPIResponse,
    summary="Run Agent 8: Scene Role Enumeration Agent",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def scene_role_enumeration_endpoint(body: SceneRoleEnumerationRequest) -> SceneRoleEnumerationAPIResponse:
    from main import _db

    try:
        result = await run_scene_role_enumeration_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Scene Role Enumeration (Agent 8) executed. Project: {body.project_id}. Status: {getattr(result, 'status', 'completed')}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }

@phase2_router.post(
    "/scene-role-selector",
    response_model=SceneRoleSelectorAPIResponse,
    summary="Run Agent 9: Scene Role Selector",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def scene_role_selector_endpoint(body: SceneRoleSelectorRequest) -> SceneRoleSelectorAPIResponse:
    """
    Triggers the Scene Role Selector Agent (Agent 9).
    Selects optimal narrative role based on evaluated pressures.
    """
    from main import _db

    try:
        result = await run_scene_role_selector_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Scene Role Selector (Agent 9) executed. Project: {body.project_id}. Status: {getattr(result, 'status', 'completed')}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }



@phase2_router.post(
    "/temporal-placement-solver",
    response_model=TemporalPlacementSolverAPIResponse,
    summary="Run Agent 10: Temporal Placement Solver",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def temporal_placement_solver_endpoint(body: TemporalPlacementSolverRequest) -> TemporalPlacementSolverAPIResponse:
    """
    Triggers the Temporal Placement Solver Agent (Agent 10).
    Determines precise timing window for a scene.
    """
    from main import _db

    try:
        result = await run_temporal_placement_solver_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Temporal Placement Solver (Agent 10) executed. Project: {body.project_id}. Status: {getattr(result, 'status', 'completed')}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }

@phase2_router.post(
    "/scene-integration-plan",
    response_model=SceneIntegrationPlanResponse,
    summary="Run Agent 11: Scene Integration Plan",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_scene_integration_plan(body: SceneIntegrationPlanRequest):
    """
    Agent 11: Scene Integration Plan
    Resolves conflicts between the scene's placement and role requirements and the overall narrative skeleton.
    """
    from main import _db

    try:
        result = await run_scene_integration_plan_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Scene Integration Plan (Agent 11) executed. Project: {body.project_id}. Status: {getattr(result, 'status', 'completed')}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }

@phase2_router.post(
    "/narrative-skeleton-generator",
    response_model=NarrativeSkeletonGeneratorResponse,
    summary="Run Agent 12: Narrative Skeleton Generator",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_narrative_skeleton_generator(body: NarrativeSkeletonGeneratorRequest):
    """
    Agent 12: Narrative Skeleton Generator
    Generates abstract story flow templates and selects the strongest skeleton.
    """
    from main import _db

    try:
        result = await run_narrative_skeleton_generator(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": f"Narrative Skeleton Generator (Agent 12) executed. Project: {body.project_id}. Status: {getattr(result, 'status', 'completed')}",
        "data": result.model_dump() if hasattr(result, "model_dump") else result
    }


@phase2_router.post(
    "/narrative-skeleton-planner",
    response_model=NarrativeSkeletonPlannerResponse,
    summary="Run Agent 13: Narrative Skeleton Planner",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_narrative_skeleton_planner(body: NarrativeSkeletonPlannerRequest):
    """
    Agent 13: Narrative Skeleton Planner
    Plans how a master narrative skeleton is mutated across 6 distinct concepts.
    """
    from main import _db

    try:
        result = await run_narrative_skeleton_planner(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    # Expecting result to be a dictionary based on run_narrative_skeleton_planner output format
    status_str = result.get('data', {}).get('status', 'completed') if isinstance(result, dict) else 'completed'

    return {
        "message": f"Narrative Skeleton Planner (Agent 13) executed. Project: {body.project_id}. Status: {status_str}",
        "data": result.get('data', {}) if isinstance(result, dict) else result
    }

@phase2_router.post(
    "/narrative-archetype-selector",
    response_model=NarrativeArchetypeSelectorResponse,
    summary="Run Agent 14: Narrative Archetype Selector",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_narrative_archetype_selector(body: NarrativeArchetypeSelectorRequest) -> NarrativeArchetypeSelectorResponse:
    """
    Agent 14: Narrative Archetype Selector
    Selects the deep emotional archetype that governs the portfolio's psychological logic.
    """
    from main import _db

    try:
        result = await run_narrative_archetype_selector(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    result_data = result.model_dump() if hasattr(result, "model_dump") else result
    status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')

    return NarrativeArchetypeSelectorResponse(
        message=f"Narrative Archetype Selector (Agent 14) executed. Project: '{body.project_id}'. Status: {status_str}",
        data=result_data
    )

@phase2_router.post(
    "/platform-behavior-optimizer",
    response_model=PlatformBehaviorOptimizerResponse,
    summary="Run Agent 15: Platform Behavior Optimizer",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_platform_behavior_optimizer(body: PlatformBehaviorOptimizerRequest) -> PlatformBehaviorOptimizerResponse:
    """
    Agent 15: Platform Behavior Optimizer
    Infers the most likely distribution platform and translates it into concept-level physics rules.
    """
    from main import _db

    try:
        result = await run_platform_behavior_optimizer_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    result_data = result.model_dump() if hasattr(result, "model_dump") else result
    status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')

    return PlatformBehaviorOptimizerResponse(
        message=f"Platform Behavior Optimizer (Agent 15) executed. Project: '{body.project_id}'. Status: {status_str}",
        data=result_data
    )

@phase2_router.post(
    "/agent-16-pattern-interrupt",
    response_model=PatternInterruptGeneratorResponse,
    summary="Run Agent 16: Pattern Interrupt Generator",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_pattern_interrupt_generator(body: PatternInterruptGeneratorRequest) -> PatternInterruptGeneratorResponse:
    """
    Agent 16: Pattern Interrupt Generator
    Generates 5-8 hook seeds by going 180 degrees against category conventions.
    """
    from main import _db

    try:
        result = await run_pattern_interrupt_generator_agent(body.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", body.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    result_data = result.model_dump() if hasattr(result, "model_dump") else result
    status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')

    return PatternInterruptGeneratorResponse(
        message=f"Pattern Interrupt Generator (Agent 16) executed. Project: '{body.project_id}'. Status: {status_str}",
        data=result_data
    )


@phase2_router.post(
    "/viral-mechanics",
    response_model=ViralMechanicsResponse,
    summary="Run Agent 17: Viral Mechanics Agent",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_viral_mechanics(req: ViralMechanicsRequest):
    logger.info(f"API Request: /viral-mechanics for project {req.project_id}")
    from main import _db
    try:
        result = await run_viral_mechanics_agent(req.project_id, _db)
        return ViralMechanicsResponse(
            message="Viral Mechanics generated successfully.",
            data=result
        )
    except ValueError as ve:
        logger.warning(f"Validation error in /viral-mechanics: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error in /viral-mechanics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@phase2_router.post(
    "/mental-model-transformer",
    response_model=MentalModelTransformerResponse,
    summary="Run Agent 18: Mental Model Transformer",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_mental_model_transformer(req: MentalModelTransformerRequest):
    logger.info(f"API Request: /mental-model-transformer for project {req.project_id}")
    from main import _db
    try:
        result = await run_mental_model_transformer(req.project_id, _db)
        
        # Serialize the Pydantic model correctly
        result_data = result.model_dump() if hasattr(result, "model_dump") else result
        status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')

        return MentalModelTransformerResponse(
            message=f"Mental Model Transformer (Agent 18) executed. Project: '{req.project_id}'. Status: {status_str}",
            data=result_data
        )
    except ValueError as ve:
        logger.warning(f"Validation error in /mental-model-transformer: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error in /mental-model-transformer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@phase2_router.post(
    "/intergalactic-thinking",
    response_model=IntergalacticThinkingResponse,
    summary="Run Agent 19: Intergalactic Thinking Agent",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_intergalactic_thinking(req: IntergalacticThinkingRequest):
    logger.info(f"API Request: /intergalactic-thinking for project {req.project_id}")
    from main import _db
    try:
        result = await run_intergalactic_thinking_agent(req.project_id, _db)
        
        # Serialize the Pydantic model correctly
        result_data = result.model_dump() if hasattr(result, "model_dump") else result
        status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')
        
        return IntergalacticThinkingResponse(
            message=f"Intergalactic Thinking Agent (Agent 19) executed. Project: '{req.project_id}'. Status: {status_str}",
            data=result_data
        )
    except ValueError as ve:
        logger.warning(f"Validation error in /intergalactic-thinking: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error in /intergalactic-thinking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@phase2_router.post(
    "/agent-20-concept-generator",
    response_model=ConceptGeneratorResponse,
    summary="Run Agent 20: Concept Generator",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_concept_generator(req: ConceptGeneratorRequest):
    logger.info(f"API Request: /agent-20-concept-generator for project {req.project_id}")
    from main import _db
    try:
        result = await run_concept_generator_agent(req.project_id, _db)

        result_data = result.model_dump() if hasattr(result, "model_dump") else result
        return ConceptGeneratorResponse(
            message=f"Concept Generator (Agent 20) executed. Project: '{req.project_id}'.",
            data=result_data
        )
    except ValueError as ve:
        logger.warning(f"Validation error in /agent-20-concept-generator: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error in /agent-20-concept-generator: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@phase2_router.post(
    "/offer-narrative-integrator",
    response_model=OfferNarrativeIntegratorResponse,
    summary="Run Agent 21: Offer Narrative Integrator",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_offer_narrative_integrator(req: OfferNarrativeIntegratorRequest) -> OfferNarrativeIntegratorResponse:
    """
    Agent 21: Offer Narrative Integrator.
    Integrates commercial offer mechanics into concept-level narrative beats.
    """
    from main import _db

    try:
        result = await run_offer_narrative_integrator_agent(req.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", req.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", req.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    result_data = result.model_dump() if hasattr(result, "model_dump") else result
    status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')

    return OfferNarrativeIntegratorResponse(
        message=f"Offer Narrative Integrator (Agent 21) executed. Project: '{req.project_id}'. Status: {status_str}",
        data=result_data,
    )


@phase2_router.post(
    "/concept-categorization",
    response_model=ConceptCategorizationResponse,
    summary="Run Agent 22: Concept Categorization Agent",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_concept_categorization(req: ConceptCategorizationRequest) -> ConceptCategorizationResponse:
    """
    Agent 22: Concept Categorization Agent.
    Assigns each concept into PITCH, PLAY, or PLUNGE and audits portfolio balance.
    """
    from main import _db

    try:
        result = await run_concept_categorization_agent(req.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", req.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", req.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    result_data = result.model_dump() if hasattr(result, "model_dump") else result
    status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')

    return ConceptCategorizationResponse(
        message=f"Concept Categorization Agent (Agent 22) executed. Project: '{req.project_id}'. Status: {status_str}",
        data=result_data,
    )


@phase2_router.post(
    "/concept-diversity-controller",
    response_model=ConceptDiversityControllerResponse,
    summary="[DEPRECATED] Agent 23: Concept Diversity Controller",
    tags=["Agents — Phase 2"],
    status_code=410,
)
async def api_concept_diversity_controller(req: ConceptDiversityControllerRequest) -> ConceptDiversityControllerResponse:
    """
    DEPRECATED. Replaced by /diversity-manifest-generator which runs before concept generation.
    The old post-generation diversity controller has been removed from the pipeline.
    """
    raise HTTPException(
        status_code=410,
        detail="concept_diversity_controller is deprecated. Use /diversity-manifest-generator instead.",
    )

    result_data = {}  # unreachable — satisfies linter
    status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')

    return ConceptDiversityControllerResponse(
        message=f"Concept Diversity Controller (Agent 23) executed. Project: '{req.project_id}'. Status: {status_str}",
        data=result_data,
    )


class VisualStructureAgentRequest(BaseModel):
    project_id: str

class VisualStructureAgentResponse(BaseModel):
    message: str
    data: dict

@phase2_router.post(
    "/visual-structure-agent",
    response_model=VisualStructureAgentResponse,
    summary="Run Visual Structure Agent (Phase 2-V)",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_visual_structure_agent(req: VisualStructureAgentRequest):
    from main import _db
    try:
        result = await run_visual_structure_agent(req.project_id, _db)
        return VisualStructureAgentResponse(
            message=f"Visual Structure Agent executed. Project: '{req.project_id}'.",
            data=result.model_dump() if hasattr(result, "model_dump") else result,
        )
    except Exception as e:
        logger.error(f"Error in /visual-structure-agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class VisualMotifSelectorRequest(BaseModel):
    project_id: str

class VisualMotifSelectorResponse(BaseModel):
    message: str
    data: dict

@phase2_router.post(
    "/visual-motif-selector",
    response_model=VisualMotifSelectorResponse,
    summary="Run Visual Motif Selector (Phase 2-V)",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_visual_motif_selector(req: VisualMotifSelectorRequest):
    from main import _db
    try:
        result = await run_visual_motif_selector(req.project_id, _db)
        return VisualMotifSelectorResponse(
            message=f"Visual Motif Selector executed. Project: '{req.project_id}'.",
            data=result.model_dump() if hasattr(result, "model_dump") else result,
        )
    except Exception as e:
        logger.error(f"Error in /visual-motif-selector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DiversityManifestRequest(BaseModel):
    project_id: str

class DiversityManifestResponse(BaseModel):
    message: str
    data: dict

@phase2_router.post(
    "/diversity-manifest-generator",
    response_model=DiversityManifestResponse,
    summary="Run Diversity Manifest Generator",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_diversity_manifest_generator(req: DiversityManifestRequest):
    from main import _db
    try:
        result = await run_diversity_manifest_generator(req.project_id, _db)
        return DiversityManifestResponse(
            message=f"Diversity Manifest Generator executed. Project: '{req.project_id}'.",
            data=result.model_dump() if hasattr(result, "model_dump") else result,
        )
    except Exception as e:
        logger.error(f"Error in /diversity-manifest-generator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class Phase2ReviewerRequest(BaseModel):
    project_id: str

class Phase2ReviewerResponse(BaseModel):
    message: str
    data: dict

@phase2_router.post(
    "/phase2-concept-reviewer",
    response_model=Phase2ReviewerResponse,
    summary="Run Phase 2 Concept Reviewer",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_phase2_concept_reviewer(req: Phase2ReviewerRequest):
    from main import _db
    try:
        result = await run_phase2_concept_reviewer(req.project_id, _db)
        return Phase2ReviewerResponse(
            message=f"Phase 2 Concept Reviewer executed. Project: '{req.project_id}'.",
            data=result.model_dump() if hasattr(result, "model_dump") else result,
        )
    except Exception as e:
        logger.error(f"Error in /phase2-concept-reviewer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@phase2_router.post(
    "/interest-filter-agent",
    response_model=InterestFilterAgentResponse,
    summary="Run Agent 24: Interest Filter Agent",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_interest_filter_agent(req: InterestFilterAgentRequest) -> InterestFilterAgentResponse:
    """
    Agent 24: Interest Filter Agent.
    Scores each concept for community-first interest and flags boring vs thumb-stopper ideas.
    """
    from main import _db

    try:
        result = await run_interest_filter_agent(req.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", req.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", req.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    result_data = result.model_dump() if hasattr(result, "model_dump") else result
    status_str = result_data.get('status', 'completed') if isinstance(result_data, dict) else getattr(result, 'status', 'completed')

    return InterestFilterAgentResponse(
        message=f"Interest Filter Agent (Agent 24) executed. Project: '{req.project_id}'. Status: {status_str}",
        data=result_data,
    )


@phase2_router.post(
    "/concept-kill-switch",
    response_model=ConceptKillSwitchResponse,
    summary="Run Agent 25: Concept Kill Switch",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def api_concept_kill_switch_agent(req: ConceptKillSwitchRequest) -> ConceptKillSwitchResponse:
    """
    Agent 25: Concept Kill Switch.
    Final hard filter. Eliminates concepts below 6.5 composite kill score, flags for regeneration 6.5-7.5, approves 7.5+.
    """
    from main import _db

    try:
        result = await run_concept_kill_switch_agent(req.project_id, _db)
    except ValueError as exc:
        logger.warning(
            "Validation error  |  project_id=%s  error=%s", req.project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Agent runtime error  |  project_id=%s  error=%s", req.project_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    result_data = result.get("data", result) if isinstance(result, dict) else (result.model_dump() if hasattr(result, "model_dump") else result)
    status_str = result.get('status', 'completed') if isinstance(result, dict) else getattr(result, 'status', 'completed')

    return ConceptKillSwitchResponse(
        message=f"Concept Kill Switch (Agent 25) executed. Project: '{req.project_id}'. Status: {status_str}",
        data=result_data,
    )


# ---------------------------------------------------------------------------
# Full Phase 2 Pipeline Endpoint
# ---------------------------------------------------------------------------

class RunPhase2PipelineRequest(BaseModel):
    project_id: str

class RunPhase2PipelineResponse(BaseModel):
    message: str
    data: dict

@phase2_router.post(
    "/run-pipeline",
    response_model=RunPhase2PipelineResponse,
    summary="Run Full Phase 2 Pipeline (all 25 agents in sequence)",
    tags=["Agents — Phase 2"],
    status_code=200,
)
async def run_phase_2_pipeline_endpoint(body: RunPhase2PipelineRequest) -> RunPhase2PipelineResponse:
    from main import _db
    try:
        result = await run_phase_2_pipeline(body.project_id, _db)
    except ValueError as exc:
        logger.warning("Pipeline validation error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Pipeline runtime error  |  project_id=%s  error=%s", body.project_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return RunPhase2PipelineResponse(
        message=f"Phase 2 pipeline {'completed' if result.get('status') == 'completed' else 'failed'} for project '{body.project_id}'.",
        data=result,
    )

