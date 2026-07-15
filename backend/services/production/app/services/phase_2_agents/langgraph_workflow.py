#!/usr/bin/env python3
"""
LangGraph Workflow Orchestrator for Phase 2 Agents
==================================================
Coordinates the 3-agent pipeline using LangGraph state management.

Flow:
Shot List → Agent 1 (Strategy) → Human Approval → Agent 2 (Image Prompts) → Agent 3 (Prompt Review) → Final Output
"""


import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

import os
from datetime import datetime
from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from backend.services.production.app.services.phase_2_agents.workflow_state import Phase2State

# Import all agents
from backend.services.production.app.services.phase_2_agents.agent_shot_strategy.shot_strategy_agent import ShotStrategyAgent
from backend.services.production.app.services.phase_2_agents.image_prompt_generator_agent import ImagePromptGeneratorAgent
from backend.services.production.app.services.phase_2_agents.agent_prompt_review.prompt_review_agent import PromptReviewAgent
from backend.services.production.app.services.phase_2_agents.shot_design_agent import ShotDesignAgent
from backend.services.production.app.services.phase_2_agents.prompt_modifier_agent import PromptModifierAgent
from backend.services.production.app.services.phase_2_agents.imagen_generator_agent import ImagenGeneratorAgent
from backend.services.production.app.services.phase_2_agents.image_reviewer_agent import ImageReviewAgent
from backend.services.production.app.services.phase_2_agents.agent_7_shot_editor import ShotEditorAgent
from backend.services.production.app.services.phase_2_agents.agent_15A.prompt_regeneration_agent import PromptRegenerationAgent
from backend.services.production.app.services.phase_2_agents.helpers.asset_library import AssetLibrary
from backend.services.production.app.config import get_shots_service
from backend.services.production.app.services.phase_2_agents.agent_16_product_reviewer.product_reviewer_agent import ProductReviewerAgent
from backend.services.production.app.services.phase_2_agents.agent_17_product_prompt_gen.product_prompt_gen_agent import ProductPromptGenAgent
from backend.services.production.app.services.phase_2_agents.agent_18_product_editor.product_editor_agent import ProductEditorAgent


def _resolve_mongo_uri() -> Optional[str]:
    """
    Resolve MongoDB URI while honoring local-mode flag.

    Returns:
        Connection string or None if not configured.

    Raises:
        ValueError: If local-mode is enabled but local URI is missing.
    """
    allow_local = os.getenv("production_ALLOW_LOCAL_MONGO", "false").lower() == "true"
    local_uri = os.getenv("production_MONGODB_URI")
    atlas_uri = os.getenv("MONGODB_ATLAS_URI")

    if allow_local:
        if local_uri:
            return local_uri
        raise ValueError(
            "production_ALLOW_LOCAL_MONGO=true but production_MONGODB_URI is not set. "
            "Please configure the local MongoDB URI in your .env file."
        )

    return local_uri or atlas_uri


def _get_product_shot_ids(state: Dict[str, Any]) -> set:
    raw_shots = state.get("shot_list_request", {}).get("shots", [])
    return {s.get("shot_id") for s in raw_shots if s.get("product_present")} if raw_shots else set()


def _log_product_review_skip(product_shot_ids: set, product_image_url: Optional[str], *, prefix: str = "") -> None:
    if product_shot_ids and not product_image_url:
        logger.warning(
            f"   → {prefix}Product shots detected ({len(product_shot_ids)}) but product_image_url is missing. "
            "Routing to Final Approval Checkpoint"
        )
    elif not product_shot_ids and product_image_url:
        logger.info(
            f"   → {prefix}product_image_url is available but no shots are marked product_present=True. "
            "Routing to Final Approval Checkpoint"
        )
    else:
        logger.info(
            f"   → {prefix}No product shots and no product_image_url. Routing to Final Approval Checkpoint"
        )


def _fetch_project_product_image_url(project_id: Optional[str]) -> Optional[str]:
    if not project_id:
        return None

    try:
        from backend.services.production.app.services.project_service import ProjectService as _PS
        project_doc = _PS().get_project(project_id)
        return project_doc.get("product_image_s3_url") if project_doc else None
    except Exception as exc:
        logger.warning(f"Phase 2: failed to fetch product_image_s3_url from project {project_id}: {exc}")
        return None


# ============================================================================
# NODE FUNCTIONS - Each agent as a LangGraph node
# ============================================================================

def agent_1_strategy_node(state: Phase2State) -> Phase2State:
    """
    Agent 1: Shot Strategy Agent Node
    Analyzes shot list and determines generation strategies
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 1: SHOT STRATEGY AGENT")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)

    # Check if Agent 1 is already completed (strategies already approved)
    if state.get("agent1_status") == "completed":
        logger.info("Agent 1 already completed - skipping execution")
        logger.info("   → Strategies were already analyzed and approved")
        logger.info("   → Skipping MongoDB save operation to avoid duplicates")
        
        # If we have annotated_shot_list, proceed to human checkpoint
        if state.get("annotated_shot_list"):
            return {
                **state,
                "current_agent": "human_approval_checkpoint",
                "pipeline_status": "running",  # Continue the pipeline
            }
        else:
            # This shouldn't happen, but handle gracefully
            logger.warning("Agent 1 marked as completed but no annotated_shot_list found")
            return {
                **state,
                "agent1_status": "failed",
                "pipeline_status": "failed",
                "error_message": "Agent 1 marked as completed but no data found",
            }

    try:
        # Initialize agent
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ShotStrategyAgent(
            model_name="gemini-3.1-pro-preview",
            temperature=0.1,
            max_tokens=8192,
            api_key=api_key
        )

        # Convert shot list request to ShotList
        from backend.services.production.app.models.mongodb.shots import ShotList, ShotItem
        
        shots = []
        for shot_data in state["shot_list_request"]["shots"]:
            shot_item = ShotItem(
                shot_id=shot_data["shot_id"],
                description=shot_data["description"],
                duration=shot_data.get("duration"),
                scene_number=shot_data.get("scene_number"),
                sequence_number=shot_data.get("sequence_number"),
                shot_style=shot_data.get("shot_style"),
                camera_movement=shot_data.get("camera_movement"),
                source_type=shot_data.get("source_type", "generated"),
                uploaded_image_id=shot_data.get("uploaded_image_id"),
                generated_image_id=shot_data.get("generated_image_id"),
                generated_video_id=shot_data.get("generated_video_id"),
                optimized_ai_notes=shot_data.get("optimized_ai_notes"),
                characters=shot_data.get("characters"),
                locations=shot_data.get("locations")
            )
            shots.append(shot_item)

        shot_list = ShotList(
            episode_id=state["episode_id"],
            title=state["title"],
            shots=shots
        )

        # Analyze shots for generation strategies
        analysis_result = agent.analyze_shot_list(shot_list)

        # Save to MongoDB if client is available
        mongodb_operations = {}
        if state.get("mongodb_client"):
            try:
                # Save annotated shots to MongoDB
                mongodb_client = state["mongodb_client"]
                success = mongodb_client.save_annotated_shots_to_atlas(
                    annotated_shots=analysis_result.annotated_shots,
                    show_id=state["show_id"],
                    episode_number=state["episode_number"],
                    episode_id=state["episode_id"],
                    title=state.get("title"),
                    scene_description=state.get("scene_description"),
                    overall_continuity_notes=analysis_result.overall_continuity_notes if hasattr(analysis_result, 'overall_continuity_notes') else None,
                    strategy_summary=analysis_result.strategy_summary if hasattr(analysis_result, 'strategy_summary') else None,
                    processing_metadata={"agent": "shot_strategy_agent", "version": "1.0"}
                )
                
                mongodb_operations["strategy_save"] = {
                    "success": success,
                    "message": f"Successfully saved shots to MongoDB" if success else "Failed to save to MongoDB"
                }
                
            except Exception as e:
                mongodb_operations["strategy_save"] = {
                    "success": False,
                    "error": str(e),
                    "message": f"Error saving to MongoDB: {str(e)}"
                }
        else:
            mongodb_operations["strategy_save"] = {
                "success": False,
                "message": "MongoDB client not configured"
            }

        # Update state
        return {
            **state,
            "annotated_shot_list": analysis_result,
            "strategy_analysis_results": analysis_result.model_dump() if hasattr(analysis_result, 'model_dump') else analysis_result,
            "agent1_status": "completed",
            "current_agent": "human_approval_checkpoint",
            "pipeline_status": "waiting_for_approval",
            "mongodb_operations": {**state.get("mongodb_operations", {}), **mongodb_operations},
            "requires_human_feedback": True,
            "feedback_agent": "agent_1",
        }

    except Exception as e:
        logger.error(f"Agent 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent1_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def human_approval_checkpoint(state: Phase2State) -> Phase2State:
    """
    Human Approval Checkpoint Node
    Pauses workflow and waits for human decision on strategies
    """
    logger.info("\n" + "="*60)
    logger.info("🧑 HUMAN APPROVAL CHECKPOINT")
    logger.info("="*60)

    # Check if human has already provided a decision
    if state.get("strategy_approval_decision") is not None:
        decision = state["strategy_approval_decision"]
        logger.info(f"✓ Human decision already received: {decision}")
        logger.info("   → Strategies were already approved/rejected")

        if decision:
            logger.info("   → Routing to Agent 2 (Image Prompt Generator)")
            return {
                **state,
                "pipeline_status": "running",
                "current_agent": "agent_2",
                "requires_human_feedback": False,
                "feedback_agent": None,
            }
        else:
            logger.info("   → Strategies rejected - ending workflow")
            return {
                **state,
                "pipeline_status": "rejected",
                "current_agent": "completed",
                "requires_human_feedback": False,
                "feedback_agent": None,
            }

    # No decision yet - pause workflow
    logger.info("⏸Workflow paused. Awaiting human approval for strategies...")
    logger.info("   - Shot strategies have been analyzed and saved to MongoDB")
    logger.info("   - Human can now review strategies and approve/reject")
    logger.info("   - Decision: 'approve' → Agent 2, 'reject' → End")
    
    return {
        **state,
        "pipeline_status": "waiting_for_approval",
        "current_agent": "human_approval_checkpoint",
        "requires_human_feedback": True,
        "feedback_agent": "agent_1",
    }


def agent_2_prompt_generator_node(state: Phase2State) -> Phase2State:
    """
    Agent 2: Image Prompt Generator Node
    Generates cinematic image prompts for each shot
    """
    logger.info("\n" + "="*60)
    logger.info("✨ AGENT 2: IMAGE PROMPT GENERATOR")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)

    try:
        # Initialize asset library using show_id (matches _id in production_projects)
        mongodb_client = state.get("mongodb_client")
        show_id = state.get("show_id")
        project_id = state.get("project_id")  # Fallback to project_id if show_id is not available

        # Log state information for debugging
        logger.info(f"State check - show_id: {show_id}, project_id: {project_id}, mongodb_client: {mongodb_client is not None}")

        # Determine which identifier to use (prefer show_id, fallback to project_id)
        identifier = show_id or project_id

        # Create AssetLibrary to fetch assets from Agent 5 and Agent 8
        if identifier and mongodb_client:
            logger.info(f"📦 Loading assets from Agent 5 and Agent 8 for identifier: {identifier} (show_id: {show_id}, project_id: {project_id})")
            asset_library = AssetLibrary(
                mongodb_client=mongodb_client,
                show_id=identifier,  # Use identifier (show_id or project_id)
                project_id=project_id  # Also pass project_id for compatibility
            )
        else:
            logger.warning(f"No identifier (show_id/project_id) or MongoDB client. Creating empty asset library.")
            logger.warning(f"  - show_id: {show_id}")
            logger.warning(f"  - project_id: {project_id}")
            logger.warning(f"  - mongodb_client: {mongodb_client is not None}")
            asset_library = AssetLibrary(mongodb_client=None)

        # Initialize agent with AssetLibrary
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ImagePromptGeneratorAgent(
            model_name="gemini-3.1-pro-preview",
            temperature=0.3,
            max_tokens=8192,
            api_key=api_key,
            asset_library=asset_library  # Pass AssetLibrary to access Agent 5 and Agent 8 assets
        )

        # Get annotated shot list from state
        annotated_list = state["annotated_shot_list"]
        if not annotated_list:
            raise ValueError("No annotated shot list found in state")

        # Build product shot context
        product_image_url = state.get("product_image_url")
        raw_shots = state.get("shot_list_request", {}).get("shots", [])
        product_shot_ids = {s.get("shot_id") for s in raw_shots if s.get("product_present")} if raw_shots else set()
        if product_shot_ids:
            logger.info(f"Agent 2: product shots detected: {product_shot_ids}")

        # Generate image prompts
        import asyncio
        annotated_list_with_prompts = asyncio.run(agent.generate_prompts_for_shots(
            annotated_list=annotated_list,
            scene_contexts=None,
            scene_description=state.get("scene_description"),
            movie_id=state.get("movie_id"),
            product_shot_ids=product_shot_ids,
            product_image_url=product_image_url
        ))

        # Update MongoDB with generated prompts using new versioned structure (no local file storage)
        mongodb_client = state.get("mongodb_client")
        if mongodb_client:
            for shot in annotated_list_with_prompts.annotated_shots:
                # Check for v0 image data in new versioned structure
                if shot.image and "v0" in shot.image:
                    v0_data = shot.image["v0"]
                    # Use new versioned structure - save as v0 (initial draft)
                    mongodb_client.update_shot_image_version(
                        show_id=state["show_id"],
                        episode_number=state["episode_number"],
                        shot_id=shot.shot_id,
                        version="v0",
                        updated_prompt=v0_data.get("updated_prompt", ""),
                        changes_made=v0_data.get("changes_made", "Initial image prompt generated by Agent 2"),
                        reasoning=v0_data.get("reasoning", "AI-generated prompt based on shot description and strategy"),
                        generated_images_s3=v0_data.get("generated_images_s3", [])
                    )

        # Update state
        # Update state
        return {
            **state,
            "image_prompts_generated": annotated_list_with_prompts.model_dump() if hasattr(annotated_list_with_prompts, 'model_dump') else annotated_list_with_prompts,
            "agent2_status": "completed",
            "current_agent": "agent_3",
        }
    
    except Exception as e:
        logger.error(f"Agent 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent2_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }
    finally:
        # Save agent 2 output to MongoDB production_projects collection
        if state.get("show_id") or state.get("project_id"):
            try:
                from backend.services.production.app.services.project_service import ProjectService
                from datetime import datetime
                
                project_id = state.get("show_id") or state.get("project_id")
                project_service = ProjectService()
                
                # Prepare output data
                try:
                    prompts_list = []
                    if 'annotated_list_with_prompts' in locals() and annotated_list_with_prompts:
                        for shot in annotated_list_with_prompts.annotated_shots:
                            prompts_list.append({
                                "shot_id": shot.shot_id,
                                "prompt": shot.image.get("v0", {}).get("updated_prompt") if shot.image else None,
                                "reasoning": shot.image.get("v0", {}).get("reasoning") if shot.image else None
                            })

                    agent2_output = {
                        "agent": "Agent 2: Image Prompt Generator",
                        "timestamp": datetime.utcnow().isoformat(),
                        "prompts": prompts_list,
                        "statistics": {
                            "total_shots": len(prompts_list),
                        }
                    }
                    
                    project_service.update_agent_output(
                        project_id=project_id,
                        agent_number=2,
                        status="completed" if 'annotated_list_with_prompts' in locals() else "failed",
                        output=agent2_output
                    )
                    logger.info(f"Agent 2 output saved to production_projects (project_id: {project_id})")
                except Exception as inner_e:
                    logger.warning(f"Failed to prepare Agent 2 output for storage: {inner_e}")
                    
            except Exception as e:
                logger.warning(f"Error saving Agent 2 output to MongoDB: {e}")


def agent_3_prompt_review_node(state: Phase2State) -> Phase2State:
    """
    Agent 3: Prompt Review Agent Node
    Reviews and refines generated image prompts for continuity
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 3: PROMPT REVIEW AGENT")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)

    try:
        # Initialize asset library using show_id (matches _id in production_projects)
        mongodb_client = state.get("mongodb_client")
        show_id = state.get("show_id")
        project_id = state.get("project_id")
        identifier = show_id or project_id

        # Create AssetLibrary to fetch assets from Agent 5 and Agent 8
        if identifier and mongodb_client:
            logger.info(f"📦 Loading assets for Agent 3 (identifier: {identifier})")
            asset_library = AssetLibrary(
                mongodb_client=mongodb_client,
                show_id=identifier,
                project_id=project_id
            )
        else:
            logger.warning(f"No identifier or MongoDB client for Agent 3. Creating empty asset library.")
            asset_library = AssetLibrary(mongodb_client=None)

        # Initialize agent
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = PromptReviewAgent(
            model_name="gemini-3.1-pro-preview",
            temperature=0.3,
            max_tokens=8192,
            api_key=api_key,
            asset_library=asset_library
        )

        # Get annotated shot list with prompts from state
        annotated_list_data = state.get("image_prompts_generated")
        if not annotated_list_data:
            # Try to get from original annotated_shot_list if image_prompts_generated is not available
            annotated_list_data = state.get("annotated_shot_list")
            if not annotated_list_data:
                raise ValueError("No image prompts found in state")

        # Convert back to AnnotatedShotList object
        from backend.services.production.app.models.mongodb.shots import AnnotatedShotList, AnnotatedShotItem
        
        # Handle both dictionary and AnnotatedShotList object cases
        if isinstance(annotated_list_data, dict):
            # If it's a dictionary (from model_dump()), convert back to AnnotatedShotList
            annotated_shots = []
            for shot_data in annotated_list_data["annotated_shots"]:
                shot = AnnotatedShotItem(**shot_data)
                annotated_shots.append(shot)
            
            annotated_list = AnnotatedShotList(
                episode_id=annotated_list_data["episode_id"],
                title=annotated_list_data["title"],
                annotated_shots=annotated_shots,
                overall_continuity_notes=annotated_list_data.get("overall_continuity_notes"),
                strategy_summary=annotated_list_data.get("strategy_summary", {})
            )
        else:
            # If it's already an AnnotatedShotList object, use it directly
            annotated_list = annotated_list_data

        # Build product shot context
        product_image_url_a3 = state.get("product_image_url")
        raw_shots_a3 = state.get("shot_list_request", {}).get("shots", [])
        product_shot_ids_a3 = {s.get("shot_id") for s in raw_shots_a3 if s.get("product_present")} if raw_shots_a3 else set()
        if product_shot_ids_a3:
            logger.info(f"Agent 3: product shots detected: {product_shot_ids_a3}")

        # Review and refine prompts
        import asyncio
        annotated_list_reviewed, review_results = asyncio.run(agent.review_prompts(
            annotated_list=annotated_list,
            scene_description=state.get("scene_description"),
            product_shot_ids=product_shot_ids_a3,
            product_image_url=product_image_url_a3
        ))

        # Update MongoDB with reviewed prompts using new versioned structure (no local file storage)
        mongodb_client = state.get("mongodb_client")
        if mongodb_client:
            for shot in annotated_list_reviewed.annotated_shots:
                # Check for v1 image data in new versioned structure
                if shot.image and "v1" in shot.image:
                    v1_data = shot.image["v1"]
                    # Use new versioned structure - save as v1 (reviewed version)
                    mongodb_client.update_shot_image_version(
                        show_id=state["show_id"],
                        episode_number=state["episode_number"],
                        shot_id=shot.shot_id,
                        version="v1",
                        updated_prompt=v1_data.get("updated_prompt", ""),
                        changes_made=v1_data.get("changes_made", "Prompt reviewed and refined by Agent 3 for continuity"),
                        reasoning=v1_data.get("reasoning", "Review agent applied continuity fixes and improvements"),
                        generated_images_s3=v1_data.get("generated_images_s3", [])
                    )

        # Update state - route to Agent 12 next
        # Update state - route to Agent 12 next
        return {
            **state,
            "reviewed_prompts": review_results,
            "annotated_shot_list": annotated_list_reviewed,  # Update with reviewed version
            "agent3_status": "completed",
            "current_agent": "agent_12",
            "pipeline_status": "running",
        }

    except Exception as e:
        logger.error(f"Agent 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent3_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }
    finally:
        # Save agent 3 output to MongoDB production_projects collection
        if state.get("show_id") or state.get("project_id"):
            try:
                from backend.services.production.app.services.project_service import ProjectService
                from datetime import datetime
                
                project_id = state.get("show_id") or state.get("project_id")
                project_service = ProjectService()
                
                # Prepare output data
                try:
                    prompts_list = []
                    # Get results from local variables if success, or state if available
                    results_to_save = locals().get('review_results')
                    
                    if results_to_save:
                        # review_results is typically a dict with 'results': [...]
                        if isinstance(results_to_save, dict) and 'results' in results_to_save:
                            for item in results_to_save['results']:
                                prompts_list.append({
                                    "shot_id": item.get("shot_id"),
                                    "original_prompt": item.get("original_prompt"),
                                    "refined_prompt": item.get("refined_prompt"),
                                    "issues": item.get("issues", [])
                                })
                        
                    agent3_output = {
                        "agent": "Agent 3: Prompt Review Agent",
                        "timestamp": datetime.utcnow().isoformat(),
                        "reviews": prompts_list,
                        "statistics": {
                            "total_reviewed": len(prompts_list),
                        }
                    }
                    
                    project_service.update_agent_output(
                        project_id=project_id,
                        agent_number=3,
                        status="completed" if results_to_save else "failed",
                        output=agent3_output
                    )
                    logger.info(f"Agent 3 output saved to production_projects (project_id: {project_id})")
                except Exception as inner_e:
                    logger.warning(f"Failed to prepare Agent 3 output for storage: {inner_e}")
                    
            except Exception as e:
                logger.warning(f"Error saving Agent 3 output to MongoDB: {e}")


def agent_12_shot_design_node(state: Phase2State) -> Phase2State:
    """
    Agent 12: Shot Design Agent Node
    Analyzes shots and selects assets for composition
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 12: SHOT DESIGN AGENT")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)

    try:
        from dataclasses import asdict
        # Initialize asset library using show_id (matches _id in production_projects)
        mongodb_client = state.get("mongodb_client")
        show_id = state.get("show_id")
        project_id = state.get("project_id")  # Fallback to project_id if show_id is not available
        
        # Log state information for debugging
        logger.info(f"State check - show_id: {show_id}, project_id: {project_id}, mongodb_client: {mongodb_client is not None}")
        
        # Recreate MongoDB client if it's missing from state (LangGraph may not preserve complex objects)
        if not mongodb_client:
            logger.warning("MongoDB client not found in state, attempting to recreate...")
            try:
                # Use singleton ShotsService instead of creating new MongoDBAtlasClient
                # This ensures only ONE DB connection is used across the entire application
                mongodb_client = get_shots_service()
                # Verify connection by checking if client and collection are available
                if hasattr(mongodb_client, 'client') and mongodb_client.client is not None:
                    logger.info("✅ Successfully retrieved ShotsService singleton")
                elif hasattr(mongodb_client, 'shots_collection') and mongodb_client.shots_collection is not None:
                    logger.info("✅ Successfully retrieved ShotsService singleton (collection available)")
                else:
                    logger.warning("ShotsService retrieved but structure unclear")
            except Exception as e:
                logger.error(f"Failed to get ShotsService: {e}")
                import traceback
                logger.error(traceback.format_exc())
                mongodb_client = None
        
        # Determine which identifier to use (prefer show_id, fallback to project_id)
        identifier = show_id or project_id
        
        # Check if mongodb_client is valid (ShotsService uses singleton connection)
        if mongodb_client:
            # ShotsService has client and shots_collection attributes
            if hasattr(mongodb_client, 'shots_collection') and mongodb_client.shots_collection is not None:
                logger.info(f"MongoDB client is valid with collection connection")
            elif hasattr(mongodb_client, 'client') and mongodb_client.client is not None:
                logger.info(f"MongoDB client structure validated")
            else:
                logger.warning(f"MongoDB client does not have expected structure. Type: {type(mongodb_client)}")
                mongodb_client = None
        
        if identifier and mongodb_client:
            logger.info(f"📦 Loading assets from agent8 output for identifier: {identifier} (show_id: {show_id}, project_id: {project_id})")
            asset_library = AssetLibrary(
                mongodb_client=mongodb_client,
                show_id=identifier,  # Use identifier (show_id or project_id)
                project_id=project_id  # Also pass project_id for compatibility
            )
        else:
            logger.warning(f"No identifier (show_id/project_id) or MongoDB client. Creating empty asset library.")
            logger.warning(f"  - show_id: {show_id}")
            logger.warning(f"  - project_id: {project_id}")
            logger.warning(f"  - mongodb_client: {mongodb_client is not None}")
            asset_library = AssetLibrary(mongodb_client=None)

        # Initialize agent
        agent = ShotDesignAgent(
            asset_library=asset_library,
            use_feasibility_check=True
        )

        # Get annotated shot list from state
        annotated_list = state.get("annotated_shot_list")
        if not annotated_list:
            raise ValueError("No annotated shot list found in state")

        # Process each shot
        shot_designs = []
        previous_shot_design = None

        logger.info(f"\nAnalyzing {len(annotated_list.annotated_shots)} shots...")

        for shot in annotated_list.annotated_shots:
            # Pass AnnotatedShotItem directly (it has characters and locations from CSV)
            # No need to convert to ShotInput - AnnotatedShotItem already has all required fields
            design_output = agent.analyze_shot(shot, previous_shot_design)
            shot_designs.append(design_output)
            
            # Store in shot history
            agent.shot_history.append(design_output)
            previous_shot_design = design_output

            # Update MongoDB with shot design
            if mongodb_client:
                design_dict = asdict(design_output)
                
                # Store shot_design data in MongoDB
                mongodb_client.shots_collection.update_one(
                    {
                        "show_id": state["show_id"],
                        "episode_number": state["episode_number"],
                        "annotated_shots.shot_id": shot.shot_id
                    },
                    {
                        "$set": {
                            "annotated_shots.$.shot_design": design_dict
                        }
                    }
                )

        logger.info(f"\nAgent 12 completed: Analyzed {len(shot_designs)} shots")

        # Save agent 12 output to MongoDB production_projects collection (no local file storage)
        if show_id:
            try:
                from backend.services.production.app.services.project_service import ProjectService
                from datetime import datetime
                
                project_service = ProjectService()
                
                # Prepare output data matching the file format
                agent12_output = {
                    "agent": "Agent 12: Shot Design Agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "shot_designs": [asdict(d) for d in shot_designs],
                    "statistics": {
                        "total_shots": len(shot_designs),
                        "avg_feasibility": sum(d.feasibility_score for d in shot_designs) / len(shot_designs) if shot_designs else 0,
                        "strategies_used": {
                            "generate_new": sum(1 for d in shot_designs if d.generation_strategy == "generate_new"),
                            "last_frame_seed": sum(1 for d in shot_designs if d.generation_strategy == "last_frame_seed"),
                            "multi_shot": sum(1 for d in shot_designs if d.generation_strategy == "multi_shot")
                        }
                    }
                }
                
                # Save to MongoDB using ProjectService
                success = project_service.update_agent_output(
                    project_id=show_id,
                    agent_number=12,
                    status="completed",
                    output=agent12_output
                )
                
                if success:
                    logger.info(f"Agent 12 output saved to MongoDB (project_id: {show_id})")
                else:
                    logger.warning(f"Failed to save Agent 12 output to MongoDB")
                    
            except Exception as e:
                logger.warning(f"Error saving Agent 12 output to MongoDB: {e}")
                import traceback
                traceback.print_exc()

        # Update state
        return {
            **state,
            "shot_designs": {
                "designs": [asdict(d) for d in shot_designs],
                "total_shots": len(shot_designs),
                "avg_feasibility": sum(d.feasibility_score for d in shot_designs) / len(shot_designs) if shot_designs else 0
            },
            "agent12_status": "completed",
            "current_agent": "agent_13",
            "pipeline_status": "running",
        }

    except Exception as e:
        logger.error(f"Agent 12 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent12_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_13_prompt_modifier_node(state: Phase2State) -> Phase2State:
    """
    Agent 13: Prompt Modifier Agent Node
    Analyzes warnings and corrects prompts
    """
    logger.info("\n" + "="*60)
    logger.info("✏AGENT 13: PROMPT MODIFIER AGENT")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)

    try:
        from dataclasses import asdict
        # Initialize asset library using show_id (matches _id in production_projects)
        mongodb_client = state.get("mongodb_client")
        show_id = state.get("show_id")
        project_id = state.get("project_id")  # Fallback to project_id if show_id is not available
        
        # Log state information for debugging
        logger.info(f"State check - show_id: {show_id}, project_id: {project_id}, mongodb_client: {mongodb_client is not None}")
        
        # Recreate MongoDB client if it's missing from state (LangGraph may not preserve complex objects)
        if not mongodb_client:
            logger.warning("MongoDB client not found in state, attempting to recreate...")
            try:
                # Use singleton ShotsService instead of creating new MongoDBAtlasClient
                # This ensures only ONE DB connection is used across the entire application
                mongodb_client = get_shots_service()
                # Verify connection by checking if client and collection are available
                if hasattr(mongodb_client, 'client') and mongodb_client.client is not None:
                    logger.info("✅ Successfully retrieved ShotsService singleton")
                elif hasattr(mongodb_client, 'shots_collection') and mongodb_client.shots_collection is not None:
                    logger.info("✅ Successfully retrieved ShotsService singleton (collection available)")
                else:
                    logger.warning("ShotsService retrieved but structure unclear")
            except Exception as e:
                logger.error(f"Failed to get ShotsService: {e}")
                import traceback
                logger.error(traceback.format_exc())
                mongodb_client = None
        
        # Determine which identifier to use (prefer show_id, fallback to project_id)
        identifier = show_id or project_id
        
        # Check if mongodb_client is valid (ShotsService uses singleton connection)
        if mongodb_client:
            # ShotsService has client and shots_collection attributes
            if hasattr(mongodb_client, 'shots_collection') and mongodb_client.shots_collection is not None:
                logger.info(f"MongoDB client is valid with collection connection")
            elif hasattr(mongodb_client, 'client') and mongodb_client.client is not None:
                logger.info(f"MongoDB client structure validated")
            else:
                logger.warning(f"MongoDB client structure unexpected, proceeding with caution")
        
        # Initialize asset library using show_id (matches _id in production_projects)
        if identifier and mongodb_client:
            logger.info(f"📦 Loading assets from agent8 output for show_id: {identifier}")
            asset_library = AssetLibrary(
                mongodb_client=mongodb_client,
                show_id=identifier
            )
        else:
            logger.warning(f"No show_id/project_id or MongoDB client. Creating empty asset library.")
            logger.warning(f"  - show_id: {show_id}, project_id: {project_id}, identifier: {identifier}")
            logger.warning(f"  - mongodb_client: {mongodb_client}")
            asset_library = AssetLibrary(mongodb_client=None)

        # Initialize agent
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Fetch visual_style from movies collection if movie_id provided
        visual_style = None
        movie_id = state.get("movie_id")
        if movie_id:
            try:
                from backend.services.production.app.config import get_mongo_factory
                from bson import ObjectId
                from backend.shared.utils.mongodb_validators import validate_object_id
                from fastapi import HTTPException
                
                mongo_factory = get_mongo_factory()
                client, movies_collection = mongo_factory.get_collection("movies")
                
                try:
                    movie_obj_id = validate_object_id(movie_id)
                except (ValueError, HTTPException) as e:
                    logger.error(f"Invalid movie_id format: {e}")
                    raise ValueError(f"Invalid movie_id format") from e
                    
                movie = movies_collection.find_one(
                    {"_id": movie_obj_id},
                    {"global_settings.visual_style": 1}
                )
                if movie:
                    visual_style = movie.get("global_settings", {}).get("visual_style")
                    logger.info(f"✓ Fetched visual_style for Agent 13: {visual_style}")
            except Exception as e:
                logger.warning(f"Failed to fetch visual_style for Agent 13: {e}")
                visual_style = None
        
        if not visual_style:
            raise ValueError("visual_style could not be fetched for Agent 13; ensure movie_id is valid and movies document has global_settings.visual_style")
        
        agent = PromptModifierAgent(
            asset_library=asset_library,
            api_key=api_key,
            model_name="gemini-3.1-pro-preview",
            visual_style=visual_style
        )

        # Get shot designs from Agent 12
        shot_designs_data = state.get("shot_designs", {})
        shot_designs = shot_designs_data.get("designs", [])

        if not shot_designs:
            raise ValueError("No shot designs found from Agent 12")

        # Establish scene baseline from first shot
        if shot_designs:
            first_shot = shot_designs[0]
            agent.scene_baseline = {
                'shot_id': first_shot['shot_id'],
                'description': first_shot['metadata'].get('original_description', ''),
                'environment': first_shot['metadata'].get('scene_environment', first_shot['metadata'].get('original_description', '')),
                'characters': first_shot['metadata'].get('characters_found', [])
            }

        # Build product shot context
        raw_shots_a13 = state.get("shot_list_request", {}).get("shots", [])
        product_shot_ids_a13 = {s.get("shot_id") for s in raw_shots_a13 if s.get("product_present")} if raw_shots_a13 else set()
        if product_shot_ids_a13:
            logger.info(f"Agent 13: product shots detected: {product_shot_ids_a13}")

        # Process each shot
        modified_shots = []

        logger.info(f"\nProcessing {len(shot_designs)} shots...")

        for shot_design in shot_designs:
            # Modify shot based on warnings
            modified = agent.modify_shot(
                shot_design,
                agent.scene_baseline,
                is_product_shot=shot_design.get("shot_id") in product_shot_ids_a13
            )
            modified_shots.append(modified)
            
            # Store in history
            agent.shot_history.append(modified)

            # Update MongoDB with prompt modifications
            if mongodb_client and identifier:
                from dataclasses import asdict
                modified_dict = asdict(modified)
                
                # Store prompt_modifications data in MongoDB
                mongodb_client.shots_collection.update_one(
                    {
                        "show_id": identifier,
                        "episode_number": state["episode_number"],
                        "annotated_shots.shot_id": modified.shot_id
                    },
                    {
                        "$set": {
                            "annotated_shots.$.prompt_modifications": modified_dict
                        }
                    }
                )

        # Calculate statistics
        total_warnings_resolved = sum(len(m.warnings_resolved) for m in modified_shots)
        total_warnings_remaining = sum(len(m.warnings_remaining) for m in modified_shots)
        avg_feasibility_improvement = sum(m.feasibility_change for m in modified_shots) / len(modified_shots) if modified_shots else 0
        shots_improved = sum(1 for m in modified_shots if m.feasibility_change > 0)
        shots_degraded = sum(1 for m in modified_shots if m.feasibility_change < 0)
        shots_unchanged = sum(1 for m in modified_shots if m.feasibility_change == 0)

        logger.info(f"\nAgent 13 completed:")
        logger.info(f"   - Processed {len(modified_shots)} shots")
        logger.info(f"   - Resolved {total_warnings_resolved} warnings")
        logger.info(f"   - Avg feasibility improvement: {avg_feasibility_improvement:+.3f}")

        # Save agent 13 output to MongoDB production_projects collection (no local file storage)
        if identifier:
            try:
                from backend.services.production.app.services.project_service import ProjectService
                from datetime import datetime
                
                project_service = ProjectService()
                
                # Prepare output data matching the file format
                agent13_output = {
                    "agent": "Agent 13: Prompt Modifier Agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "modified_shots": [asdict(m) for m in modified_shots],
                    "statistics": {
                        "total_shots": len(modified_shots),
                        "total_warnings_resolved": total_warnings_resolved,
                        "total_warnings_remaining": total_warnings_remaining,
                        "avg_feasibility_improvement": avg_feasibility_improvement,
                        "shots_improved": shots_improved,
                        "shots_degraded": shots_degraded,
                        "shots_unchanged": shots_unchanged
                    }
                }
                
                # Save to MongoDB using ProjectService
                success = project_service.update_agent_output(
                    project_id=identifier,
                    agent_number=13,
                    status="completed",
                    output=agent13_output
                )
                
                if success:
                    logger.info(f"Agent 13 output saved to MongoDB (project_id: {identifier})")
                else:
                    logger.warning(f"Failed to save Agent 13 output to MongoDB")
                    
            except Exception as e:
                logger.warning(f"Error saving Agent 13 output to MongoDB: {e}")
                import traceback
                traceback.print_exc()

        # Update state - route to prompt approval checkpoint
        return {
            **state,
            "modified_prompts": {
                "modified_shots": [asdict(m) for m in modified_shots],
                "total_shots": len(modified_shots),
                "total_warnings_resolved": total_warnings_resolved,
                "avg_feasibility_improvement": avg_feasibility_improvement
            },
            "agent13_status": "completed",
            "prompt_approval_decision": True,
            "current_agent": "agent_14",
            "pipeline_status": "running",
            "requires_human_feedback": False,
            "feedback_agent": None,
        }

    except Exception as e:
        logger.error(f"Agent 13 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent13_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def prompt_approval_checkpoint(state: Phase2State) -> Phase2State:
    """
    Prompt Approval Checkpoint Node
    Pauses workflow and waits for human decision on corrected prompts
    """
    logger.info("\n" + "="*60)
    logger.info("🧑 PROMPT APPROVAL CHECKPOINT")
    logger.info("="*60)

    # Check if human has already provided a decision
    if state.get("prompt_approval_decision") is not None:
        decision = state["prompt_approval_decision"]
        logger.info(f"✓ Human decision already received: {decision}")
        
        if decision:
            logger.info("   → Routing to Agent 14 (Imagen Generator)")
            return {
                **state,
                "pipeline_status": "running",
                "current_agent": "agent_14",
                "requires_human_feedback": False,
                "feedback_agent": None,
            }
        else:
            logger.info("   → Prompts rejected - ending workflow")
            return {
                **state,
                "pipeline_status": "rejected",
                "current_agent": "completed",
                "requires_human_feedback": False,
                "feedback_agent": None,
            }

    # No decision yet - pause workflow
    logger.info("⏸Workflow paused. Awaiting human approval for prompts...")
    logger.info("   - Review corrected prompts and approve/reject")
    logger.info("   - Decision: 'approve' → Agent 14, 'reject' → End")
    
    return {
        **state,
        "pipeline_status": "waiting_for_prompt_approval",
        "current_agent": "prompt_approval_checkpoint",
        "requires_human_feedback": True,
        "feedback_agent": "agent_13",
    }


def agent_14_imagen_generator_node(state: Phase2State) -> Phase2State:
    """
    Agent 14: Imagen Generator Node
    Generates images using Vertex AI Imagen 4.0 based on Agent 13's corrected prompts
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 14: IMAGEN GENERATOR")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)

    try:
        # Initialize agent with S3 client
        agent = ImagenGeneratorAgent()

        # Ensure MongoDB client exists (LangGraph may drop complex objects between nodes)
        mongodb_client = state.get("mongodb_client")
        if not mongodb_client:
            logger.warning("MongoDB client missing in state for Agent 14. Attempting to recreate...")
            try:
                mongodb_client = get_shots_service()
                logger.info("MongoDB ShotsService recreated for Agent 14")
            except Exception as mongo_error:
                logger.warning(f"Failed to recreate MongoDB client for Agent 14: {mongo_error}")
                mongodb_client = None

        # Get modified prompts from Agent 13
        modified_prompts_data = state.get("modified_prompts", {})
        modified_shots = modified_prompts_data.get("modified_shots", [])

        if not modified_shots:
            raise ValueError("No modified prompts found from Agent 13")

        # Inject product image into corrected_assets for shots where product_present=True
        product_image_url = state.get("product_image_url")
        if product_image_url:
            # Build a set of shot_ids that require the product
            raw_shots = state.get("shot_list_request", {}).get("shots", [])
            product_shot_ids = {
                s.get("shot_id") for s in raw_shots if s.get("product_present")
            }
            if product_shot_ids:
                product_asset_entry = {
                    "name": "PRODUCT",
                    "type": "product",
                    "url": product_image_url,
                }
                injected = 0
                for shot in modified_shots:
                    if shot.get("shot_id") in product_shot_ids:
                        assets = list(shot.get("corrected_assets") or [])
                        # Avoid duplicates if somehow already present
                        if not any(a.get("type") == "product" for a in assets):
                            assets.insert(0, product_asset_entry)
                            shot["corrected_assets"] = assets
                            injected += 1
                logger.info(
                    f"✓ Agent 14: injected product image reference into {injected} shot(s)"
                )
            else:
                logger.warning(
                    "⚠️  product_image_url is set but no shots have product_present=True — "
                    "product image will NOT be injected. Ensure shot documents have "
                    "product_present=True for product shots."
                )

        # Get movie_id for aspect ratio lookup from movies collection
        movie_id = state.get("movie_id")
        show_id = state.get("show_id")

        # Generate images for all shots
        results = agent.generate_images_for_shots(
            modified_shots=modified_shots,
            output_dir="backend/services/production/app/services/phase_2_agents/outputs/agent_14_generated_images",
            movie_id=movie_id
        )

        # Save metadata to file
        metadata_path = agent.save_metadata()

        logger.info(f"\nAgent 14 completed: Generated {len(results['generated_images'])} images")

        # Save agent 14 output to MongoDB production_projects collection
        show_id = state.get("show_id")
        if show_id:
            try:
                from backend.services.production.app.services.project_service import ProjectService
                from dataclasses import asdict
                
                project_service = ProjectService()
                
                # Prepare output data
                agent14_output = {
                    "agent": "Agent 14: Imagen Generator",
                    "timestamp": datetime.utcnow().isoformat(),
                    "generated_images": results['generated_images'],
                    "failed_generations": results['failed_generations'],
                    "statistics": {
                        "total_shots_processed": len(results['generated_images']) + len(results['failed_generations']),
                        "successful_generations": len(results['generated_images']),
                        "failed_generations": len(results['failed_generations']),
                        "success_rate": f"{(len(results['generated_images']) / (len(results['generated_images']) + len(results['failed_generations'])) * 100):.1f}%" if (len(results['generated_images']) + len(results['failed_generations'])) > 0 else "0%"
                    }
                }
                
                # Save to MongoDB using ProjectService
                success = project_service.update_agent_output(
                    project_id=show_id,
                    agent_number=14,
                    status="completed",
                    output=agent14_output
                )
                
                if success:
                    logger.info(f"Agent 14 output saved to MongoDB (project_id: {show_id})")
                else:
                    logger.warning(f"Failed to save Agent 14 output to MongoDB")
                    
            except Exception as e:
                logger.warning(f"Error saving Agent 14 output to MongoDB: {e}")
                import traceback
                traceback.print_exc()
        
        # Save generated image S3 URLs to shots collection (using show_id for lookup)
        episode_number = state.get("episode_number", 0)
        modified_shots_map = {shot.get("shot_id"): shot for shot in modified_shots}
        
        if mongodb_client:
            for img_data in results.get("generated_images", []):
                shot_id = img_data.get('shot_id', '')
                s3_url = img_data.get('s3_url', '')
                
                if not (shot_id and s3_url):
                    continue
                
                shot_meta = modified_shots_map.get(shot_id, {})
                updated_prompt = shot_meta.get('corrected_prompt', img_data.get('prompt', ''))
                changes_made = "Initial Imagen generation (Agent 14)"
                reasoning = "Generated from Agent 13 corrected prompt"
                
                success = mongodb_client.update_shot_image_version(
                    show_id=show_id,
                    episode_number=episode_number,
                    shot_id=shot_id,
                    version="v0",
                    updated_prompt=updated_prompt or "",
                    changes_made=changes_made,
                    reasoning=reasoning,
                    generated_images_s3=[s3_url]
                )
                
                if success:
                    logger.info(f"Saved Agent 14 image (v0) to MongoDB for {shot_id}")
                else:
                    logger.warning(f"Failed to persist Agent 14 image (v0) for {shot_id}")
                
                img_data['image_version'] = "v0"

        # Update state
        return {
            **state,
            "generated_images": results,
            "agent14_status": "completed",
            "current_agent": "agent_15",
            "pipeline_status": "running",
            "output_files": state.get("output_files", []) + [metadata_path],
            "mongodb_client": mongodb_client or state.get("mongodb_client"),
        }

    except Exception as e:
        logger.error(f"Agent 14 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent14_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_14_regenerate_node(state: Phase2State) -> Phase2State:
    """
    Agent 14: Imagen Generator Node (Regeneration Path)
    Generates images using regenerated prompts from Agent 15A
    Uploads to S3 with versioning (v1, v2, ...) and updates shots_collection
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 14: IMAGEN GENERATOR (REGENERATION)")
    logger.info("Part of agent_15-15A-14-15 REGENERATION LOOP")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)

    try:
        # Initialize agent with S3 client
        agent = ImagenGeneratorAgent()

        # Get regenerated prompts from Agent 15A
        regenerated_prompts_data = state.get("regenerated_prompts", {})
        regenerated_shots = regenerated_prompts_data.get("regenerated_shots", [])

        if not regenerated_shots:
            raise ValueError("No regenerated prompts found from Agent 15A")

        # Get movie_id for aspect ratio lookup from movies collection
        movie_id = state.get("movie_id")
        show_id = state.get("show_id")
        episode_number = state.get("episode_number", 0)
        mongodb_client = state.get("mongodb_client")

        if not mongodb_client:
            logger.warning("MongoDB client missing in state for Agent 14 regeneration. Attempting to recreate...")
            try:
                mongodb_client = get_shots_service()
                logger.info("MongoDB ShotsService recreated for Agent 14 (regeneration)")
            except Exception as mongo_error:
                logger.warning(f"Failed to recreate MongoDB client for Agent 14 (regeneration): {mongo_error}")
                mongodb_client = None
        
        # Get original modified_prompts from Agent 13 to retrieve assets and metadata
        modified_prompts_data = state.get("modified_prompts", {})
        modified_prompts = modified_prompts_data.get("modified_shots", [])

        # Create a map for quick lookup of original data by shot_id
        original_data_map = {}
        for modified_shot in modified_prompts:
            shot_id = modified_shot.get('shot_id')
            if shot_id:
                original_data_map[shot_id] = modified_shot

        # Convert regenerated prompts to modified_shots format for agent_14
        modified_shots_for_generation = []
        regenerated_metadata_map = {}
        for regen_result in regenerated_shots:
            shot_id = regen_result.get('shot_id')
            if not shot_id:
                continue
            regenerated_metadata_map[shot_id] = regen_result

            # Fetch original data (assets, metadata) from Agent 13's output
            original_shot_data = original_data_map.get(shot_id, {})
            corrected_assets = original_shot_data.get('corrected_assets', [])
            metadata = original_shot_data.get('metadata', {})

            logger.info(f"Using {len(corrected_assets)} assets from Agent 13 for {shot_id}")

            modified_shots_for_generation.append({
                'shot_id': shot_id,
                'corrected_prompt': regen_result.get('updated_prompt'),
                'corrected_assets': corrected_assets,
                'metadata': metadata
            })

        # Inject product image into corrected_assets for shots where product_present=True
        product_image_url = state.get("product_image_url")
        if product_image_url:
            raw_shots = state.get("shot_list_request", {}).get("shots", [])
            product_shot_ids = {
                s.get("shot_id") for s in raw_shots if s.get("product_present")
            }
            if product_shot_ids:
                product_asset_entry = {
                    "name": "PRODUCT",
                    "type": "product",
                    "url": product_image_url,
                }
                for shot in modified_shots_for_generation:
                    if shot.get("shot_id") in product_shot_ids:
                        assets = list(shot.get("corrected_assets") or [])
                        if not any(a.get("type") == "product" for a in assets):
                            assets.insert(0, product_asset_entry)
                            shot["corrected_assets"] = assets
            else:
                logger.warning(
                    "⚠️  product_image_url is set but no shots have product_present=True — "
                    "product image will NOT be injected. Ensure shot documents have "
                    "product_present=True for product shots."
                )

        # Generate images for all shots
        results = agent.generate_images_for_shots(
            modified_shots=modified_shots_for_generation,
            output_dir="backend/services/production/app/services/phase_2_agents/outputs/agent_14_generated_images",
            movie_id=movie_id
        )

        # Save metadata to file
        metadata_path = agent.save_metadata()

        logger.info(f"\nAgent 14 (Regeneration) completed: Generated {len(results['generated_images'])} images")

        # Upload to S3 with versioned paths and update MongoDB
        generated_images_with_s3 = []
        
        # Use the agent's S3 client which is already properly configured with credential fallbacks
        s3_client = agent.s3_client
        s3_bucket = agent.s3_bucket
        
        if not s3_client:
            logger.error("S3 client not available from ImagenGeneratorAgent, attempting to create one...")
            # Fallback: try to create one using the same method as the agent
            s3_client = agent._create_s3_client()
            if not s3_client:
                logger.error("Failed to create S3 client, skipping S3 uploads")
        
        for img_data in results.get("generated_images", []):
            shot_id = img_data.get('shot_id', '')
            
            if not shot_id:
                continue
            
            # Determine next version number for image field
            current_version = 0
            if mongodb_client:
                try:
                    query = {"show_id": show_id, "annotated_shots": {"$exists": True}}
                    raw_doc = mongodb_client.shots_collection.find_one(query)
                    
                    if raw_doc and "annotated_shots" in raw_doc:
                        for shot in raw_doc["annotated_shots"]:
                            if shot.get("shot_id") == shot_id:
                                image_versions = shot.get("image")
                                if image_versions and isinstance(image_versions, dict):
                                    versions = [v for v in image_versions.keys() if v.startswith('v') and v[1:].isdigit()]
                                    if versions:
                                        version_nums = [int(v[1:]) for v in versions]
                                        current_version = max(version_nums)
                                break
                except Exception as e:
                    logger.warning(f"Error checking image versions for {shot_id}: {e}")
            
            next_version = current_version + 1
            version_key = f"v{next_version}"
            
            # Get image bytes from local path
            local_path = img_data.get('local_path', '')
            if not local_path or not os.path.exists(local_path):
                logger.warning(f"Local image not found for {shot_id}, skipping S3 upload")
                continue
            
            # Read image
            with open(local_path, 'rb') as f:
                image_bytes = f.read()
            
            # Upload to S3 with versioned path: phase2/<show_id>/<shot_id>/v{n}.png
            s3_key = f"phase2/{show_id}/{shot_id}/{version_key}.png"
            
            if s3_client:
                try:
                    s3_client.put_object(
                        Bucket=s3_bucket,
                        Key=s3_key,
                        Body=image_bytes,
                        ContentType='image/png'
                    )
                    
                    # Generate pre-signed URL (7 days) — bucket blocks public access
                    s3_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': s3_bucket, 'Key': s3_key},
                        ExpiresIn=604800
                    )
                    
                    # Update img_data with S3 URL
                    img_data['s3_url'] = s3_url
                    generated_images_with_s3.append(img_data)
                    
                    logger.info(f"Uploaded {shot_id} to S3: {s3_url}")
                    
                    # Update MongoDB shots_collection image.{version} with new data
                    if mongodb_client:
                        try:
                            regen_meta = regenerated_metadata_map.get(shot_id, {})
                            success = mongodb_client.update_shot_image_version(
                                show_id=show_id,
                                episode_number=episode_number,
                                shot_id=shot_id,
                                version=version_key,
                                updated_prompt=regen_meta.get('updated_prompt', img_data.get('prompt', '')),
                                changes_made=f"Regeneration loop iteration {next_version} via Agent 15A",
                                reasoning=regen_meta.get('reasoning', regen_meta.get('analysis', 'Agent 15A regeneration')),
                                generated_images_s3=[s3_url]
                            )
                            
                            if success:
                                logger.info(f"Updated MongoDB image.{version_key} for {shot_id}")
                            else:
                                logger.warning(f"Failed to update MongoDB for {shot_id}")
                            
                            img_data['image_version'] = version_key
                        except Exception as e:
                            logger.warning(f"Error updating MongoDB for {shot_id}: {e}")
                    
                except Exception as e:
                    logger.warning(f"Failed to upload {shot_id} to S3: {e}")
                    continue
            else:
                logger.warning(f"S3 client not available, skipping upload for {shot_id}")

        # Update results with S3 URLs
        results['generated_images'] = generated_images_with_s3

        # Update regenerate loop iterations
        regenerate_loop_iterations = state.get("regenerate_loop_iterations", {})
        shots_needing_regeneration = state.get("shots_needing_regeneration", [])
        new_regenerate_loop_iterations = {**regenerate_loop_iterations}
        for shot_id in shots_needing_regeneration:
            new_regenerate_loop_iterations[shot_id] = new_regenerate_loop_iterations.get(shot_id, 0) + 1
        
        logger.info(f"\nUpdated regenerate_loop_iterations:")
        for shot_id, count in new_regenerate_loop_iterations.items():
            logger.info(f"   {shot_id}: {count} regeneration(s)")

        # Save agent 14 output to MongoDB production_projects collection
        if show_id:
            try:
                from backend.services.production.app.services.project_service import ProjectService
                from dataclasses import asdict
                from bson import ObjectId
                from backend.services.production.app.config import get_projects_collection
                
                project_service = ProjectService()
                
                # Prepare output data
                agent14_output = {
                    "agent": "Agent 14: Imagen Generator (Regeneration)",
                    "timestamp": datetime.utcnow().isoformat(),
                    "generated_images": generated_images_with_s3,
                    "failed_generations": results.get('failed_generations', []),
                    "statistics": {
                        "total_shots_processed": len(generated_images_with_s3) + len(results.get('failed_generations', [])),
                        "successful_generations": len(generated_images_with_s3),
                        "failed_generations": len(results.get('failed_generations', [])),
                        "success_rate": f"{(len(generated_images_with_s3) / (len(generated_images_with_s3) + len(results.get('failed_generations', []))) * 100):.1f}%" if (len(generated_images_with_s3) + len(results.get('failed_generations', []))) > 0 else "0%"
                    }
                }
                
                # Save to MongoDB - append to existing agent14 output or create new
                client, projects_col = get_projects_collection()
                
                try:
                    from backend.shared.utils.mongodb_validators import validate_object_id
                    from fastapi import HTTPException
                    show_id_obj = validate_object_id(show_id)
                except (ValueError, HTTPException) as e:
                    logger.error(f"Invalid show_id format: {e}")
                    raise ValueError(f"Invalid show_id format") from e
                    
                try:
                    result = projects_col.update_one(
                        {"_id": show_id_obj},
                        {
                            "$set": {
                                "agent_outputs.agent14_regenerate.status": "completed",
                                "agent_outputs.agent14_regenerate.executed_at": datetime.utcnow(),
                                "agent_outputs.agent14_regenerate.output": agent14_output,
                                "updated_at": datetime.utcnow()
                            }
                        }
                    )
                    success = result.modified_count > 0
                    if success:
                        logger.info(f"Agent 14 (Regeneration) output saved to MongoDB")
                except Exception as e:
                    logger.warning(f"Error saving Agent 14 (Regeneration) output: {e}")
                    
            except Exception as e:
                logger.warning(f"Error saving Agent 14 (Regeneration) output to MongoDB: {e}")
                import traceback
                traceback.print_exc()

        # Merge regenerated images into existing generated_images — do NOT overwrite.
        # Overwriting would erase URLs for shots approved in earlier passes, making
        # agent 16 blind to them ("No image URL for X, force-passing").
        existing_gen = state.get("generated_images", {})
        existing_list = existing_gen.get("generated_images", [])
        new_list = results.get("generated_images", [])
        new_shot_ids = {img["shot_id"] for img in new_list if img.get("shot_id")}
        merged_list = [img for img in existing_list if img.get("shot_id") not in new_shot_ids] + new_list
        merged_generated_images = {**existing_gen, "generated_images": merged_list}

        # Update state - route back to agent_15 for review
        return {
            **state,
            "generated_images": merged_generated_images,
            "agent14_status": "completed",
            "current_agent": "agent_15",
            "pipeline_status": "running",
            "output_files": state.get("output_files", []) + [metadata_path],
            "regenerate_loop_iterations": new_regenerate_loop_iterations,
            "mongodb_client": mongodb_client or state.get("mongodb_client"),
        }

    except Exception as e:
        logger.error(f"Agent 14 (Regeneration) failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent14_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_15A_prompt_regeneration_node(state: Phase2State) -> Phase2State:
    """
    Agent 15A: Prompt Regeneration Node
    Regenerates prompts for shots that need regeneration (not editing)
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 15A: PROMPT REGENERATION AGENT")
    logger.info("Part of agent_15-15A-14-15 REGENERATION LOOP")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)
    
    try:
        # Initialize agent
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = PromptRegenerationAgent(api_key=api_key)
        
        # Get shots needing regeneration
        shots_needing_regeneration = state.get("shots_needing_regeneration", [])
        
        logger.info(f"Processing {len(shots_needing_regeneration)} shot(s) for prompt regeneration:")
        
        if not shots_needing_regeneration:
            logger.warning("No shots need regeneration")
            return {
                **state,
                "agent15A_status": "completed",
                "current_agent": "final_approval_checkpoint",
                "pipeline_status": "waiting_for_final_approval",
            }
        
        # Get review results from Agent 15
        image_reviews = state.get("image_reviews", {})
        reviews = image_reviews.get("reviews", [])
        
        # Get shot designs and modified prompts for metadata
        shot_designs_data = state.get("shot_designs", {})
        shot_designs = shot_designs_data.get("designs", [])
        
        modified_prompts_data = state.get("modified_prompts", {})
        modified_prompts = modified_prompts_data.get("modified_shots", [])
        
        # Get generated images to find S3 URLs and original prompts
        generated_images = state.get("generated_images", {}).get("generated_images", [])
        
        # Get current regenerate loop iterations
        regenerate_loop_iterations = state.get("regenerate_loop_iterations", {})
        
        # Prepare shots data for regeneration
        shots_to_regenerate = []
        for shot_id in shots_needing_regeneration:
            # SAFEGUARD: Check if shot has already reached max regeneration iterations (3)
            current_regenerate_iteration = regenerate_loop_iterations.get(shot_id, 0)
            if current_regenerate_iteration >= 3:
                logger.warning(f"Skipping {shot_id} - already at max regeneration iterations (3). Current iteration: {current_regenerate_iteration}")
                continue
            
            # Find review for this shot
            review = next((r for r in reviews if r.get('shot_id') == shot_id), None)
            if not review:
                logger.warning(f"No review found for shot {shot_id}")
                continue
            
            # Find original prompt - check modified_prompts first (from agent_13)
            older_prompt = None
            prompt_data = next((p for p in modified_prompts if p.get('shot_id') == shot_id), None)
            if prompt_data:
                older_prompt = prompt_data.get('corrected_prompt', '')
            
            # Fallback: get from generated_images metadata
            if not older_prompt:
                for img in generated_images:
                    if img.get('shot_id') == shot_id:
                        older_prompt = img.get('prompt', '')
                        break
            
            # Get current S3 URL from generated_images or MongoDB
            current_s3_url = None
            for img in generated_images:
                if img.get('shot_id') == shot_id:
                    current_s3_url = img.get('s3_url', '')
                    break
            
            # If not found in state, try MongoDB
            mongodb_client = state.get("mongodb_client")
            if not current_s3_url and mongodb_client:
                try:
                    query = {"show_id": state["show_id"], "annotated_shots": {"$exists": True}}
                    raw_doc = mongodb_client.shots_collection.find_one(query)
                    
                    if raw_doc and "annotated_shots" in raw_doc:
                        for shot in raw_doc["annotated_shots"]:
                            if shot.get("shot_id") == shot_id:
                                # Check edited_image_s3 for latest version
                                edited_image_s3 = shot.get("edited_image_s3")
                                if edited_image_s3 and isinstance(edited_image_s3, dict):
                                    versions = list(edited_image_s3.keys())
                                    if versions:
                                        versions.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0, reverse=True)
                                        latest_version = versions[0]
                                        latest_image_data = edited_image_s3[latest_version]
                                        current_s3_url = latest_image_data.get('s3_url', '')
                                # Fallback to generated_image_s3
                                if not current_s3_url:
                                    current_s3_url = shot.get('generated_image_s3', '')
                                break
                except Exception as e:
                    logger.warning(f"Error fetching from MongoDB: {e}")
            
            if not current_s3_url:
                logger.warning(f"No S3 URL found for shot {shot_id}, skipping")
                continue
            
            if not older_prompt:
                logger.warning(f"No older prompt found for shot {shot_id}, skipping")
                continue
            
            # Format edit_instructions - convert issues_found list to string if needed
            edit_instructions = review.get('edit_instructions', '')
            if not edit_instructions:
                issues_found = review.get('issues_found', [])
                if isinstance(issues_found, list):
                    edit_instructions = "; ".join([issue.get('description', '') if isinstance(issue, dict) else str(issue) for issue in issues_found])
                else:
                    edit_instructions = str(issues_found) if issues_found else "Image requires regeneration"
            
            shots_to_regenerate.append({
                'shot_id': shot_id,
                'image_s3_url': current_s3_url,
                'older_prompt': older_prompt,
                'edit_instructions': edit_instructions
            })
        
        # Build product shot context
        raw_shots_a15a = state.get("shot_list_request", {}).get("shots", [])
        product_shot_ids_a15a = {s.get("shot_id") for s in raw_shots_a15a if s.get("product_present")} if raw_shots_a15a else set()
        if product_shot_ids_a15a:
            logger.info(f"Agent 15A: product shots detected: {product_shot_ids_a15a}")

        # Regenerate prompts
        regeneration_results = agent.regenerate_prompts_batch(
            shots_to_regenerate=shots_to_regenerate,
            shot_designs=shot_designs,
            modified_prompts=modified_prompts,
            product_shot_ids=product_shot_ids_a15a
        )
        
        logger.info(f"\nAgent 15A completed: Regenerated prompts for {len(regeneration_results)} shots")
        
        # Save regeneration results to file
        from backend.services.production.app.services.phase_2_agents.agent_15A.prompt_regeneration_agent import save_results
        try:
            saved_file = save_results(regeneration_results, "backend/services/production/app/services/phase_2_agents/outputs/agent_15A")
            logger.info(f"Regeneration results saved to: {saved_file}")
        except Exception as e:
            logger.warning(f"Failed to save regeneration results: {e}")
        
        # Save agent 15A output to MongoDB production_projects collection
        show_id = state.get("show_id")
        if show_id:
            try:
                from backend.services.production.app.services.project_service import ProjectService
                from dataclasses import asdict
                from bson import ObjectId
                from backend.services.production.app.config import get_projects_collection
                
                project_service = ProjectService()
                
                # Prepare output data
                agent15A_output = {
                    "agent": "Agent 15A: Prompt Regeneration Agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "regenerated_prompts": [asdict(result) for result in regeneration_results],
                    "statistics": {
                        "total_shots_processed": len(regeneration_results),
                        "successful_regenerations": len([r for r in regeneration_results if r.updated_prompt != r.older_prompt]),
                        "failed_regenerations": len([r for r in regeneration_results if r.updated_prompt == r.older_prompt])
                    }
                }
                
                # Save to MongoDB - directly update production_projects with agent15A key
                client, projects_col = get_projects_collection()
                
                try:
                    from backend.shared.utils.mongodb_validators import validate_object_id
                    from fastapi import HTTPException
                    show_id_obj = validate_object_id(show_id)
                except (ValueError, HTTPException) as e:
                    logger.error(f"Invalid show_id format: {e}")
                    raise ValueError(f"Invalid show_id format") from e
                    
                try:
                    result = projects_col.update_one(
                        {"_id": show_id_obj},
                        {
                            "$set": {
                                "agent_outputs.agent15A.status": "completed",
                                "agent_outputs.agent15A.executed_at": datetime.utcnow(),
                                "agent_outputs.agent15A.output": agent15A_output,
                                "updated_at": datetime.utcnow()
                            }
                        }
                    )
                    success = result.modified_count > 0
                except Exception as e:
                    logger.warning(f"Error directly updating MongoDB: {e}")
                    success = False
                
                if success:
                    logger.info(f"Agent 15A output saved to MongoDB (project_id: {show_id})")
                else:
                    logger.warning(f"Failed to save Agent 15A output to MongoDB")
                    
            except Exception as e:
                logger.warning(f"Error saving Agent 15A output to MongoDB: {e}")
                import traceback
                traceback.print_exc()
        
        # Update state - route to agent_14 for regeneration
        return {
            **state,
            "regenerated_prompts": {
                "regenerated_shots": [asdict(result) for result in regeneration_results],
                "total_shots": len(regeneration_results)
            },
            "agent15A_status": "completed",
            "current_agent": "agent_14_regenerate",
            "pipeline_status": "running",
        }
        
    except Exception as e:
        logger.error(f"Agent 15A failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent15A_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_15_image_reviewer_node(state: Phase2State) -> Phase2State:
    """
    Agent 15: Image Reviewer Node
    Reviews generated images using Gemini vision API
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 15: IMAGE REVIEWER")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)

    try:
        from dataclasses import asdict
        # Initialize agent
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ImageReviewAgent(api_key=api_key)

        # Get generated images from Agent 14
        generated_images_data = state.get("generated_images", {})
        generated_images = generated_images_data.get("generated_images", [])

        if not generated_images:
            raise ValueError("No generated images found from Agent 14")

        # Get shot designs from Agent 12 and modified prompts from Agent 13
        shot_designs_data = state.get("shot_designs", {})
        shot_designs = shot_designs_data.get("designs", [])
        
        modified_prompts_data = state.get("modified_prompts", {})
        modified_prompts = modified_prompts_data.get("modified_shots", [])

        # Build product shot context
        raw_shots_a15 = state.get("shot_list_request", {}).get("shots", [])
        product_shot_ids_a15 = {s.get("shot_id") for s in raw_shots_a15 if s.get("product_present")} if raw_shots_a15 else set()
        if product_shot_ids_a15:
            logger.info(f"Agent 15: product shots detected: {product_shot_ids_a15}")

        # Review all images
        summary = agent.review_all_images(
            generated_images=generated_images,
            shot_designs=shot_designs,
            modified_prompts=modified_prompts,
            product_shot_ids=product_shot_ids_a15
        )

        # Save review report
        report_path = agent.save_review_report(summary)

        logger.info(f"\nAgent 15 completed: Reviewed {summary.total_images} images")
        logger.info(f"   - Approved: {summary.approved}")
        logger.info(f"   - Edit required: {summary.edit}")
        logger.info(f"   - Regenerate required: {summary.regenerate}")

        # Save agent 15 output to MongoDB production_projects collection
        show_id = state.get("show_id")
        episode_number = state.get("episode_number", 0)
        mongodb_client = state.get("mongodb_client")

        if show_id:
            try:
                from backend.services.production.app.services.project_service import ProjectService

                project_service = ProjectService()
                
                # Prepare output data
                agent15_output = {
                    "agent": "Agent 15: Image Reviewer",
                    "timestamp": datetime.utcnow().isoformat(),
                    "summary": {
                        "total_images": summary.total_images,
                        "approved": summary.approved,
                        "edit": summary.edit,
                        "regenerate": summary.regenerate,
                        "critical_issues": summary.critical_issues
                    },
                    "reviews": [asdict(review) for review in summary.reviews]
                }
                
                # Save to MongoDB using ProjectService
                success = project_service.update_agent_output(
                    project_id=show_id,
                    agent_number=15,
                    status="completed",
                    output=agent15_output
                )
                
                if success:
                    logger.info(f"Agent 15 output saved to MongoDB (project_id: {show_id})")
                else:
                    logger.warning(f"Failed to save Agent 15 output to MongoDB")
                    
            except Exception as e:
                logger.warning(f"Error saving Agent 15 output to MongoDB: {e}")
                import traceback
                traceback.print_exc()
        
        # Process reviews and identify shots needing edit vs regeneration
        # Start with existing lists from state (preserve across iterations)
        existing_shots_needing_edit = state.get("shots_needing_edit", [])
        existing_shots_needing_regeneration = state.get("shots_needing_regeneration", [])
        
        # Get existing edit instructions dict (stores edit instructions per shot for Agent 7)
        shots_edit_instructions = state.get("shots_edit_instructions", {})
        
        # Create new lists for this review iteration
        new_shots_needing_edit = []
        new_shots_needing_regeneration = []
        shots_approved = []
        
        # Track edit loop iterations if not already initialized
        edit_loop_iterations = state.get("edit_loop_iterations", {})
        regenerate_loop_iterations = state.get("regenerate_loop_iterations", {})
        
        logger.info(f"\nAgent 15 - Processing shots:")
        logger.info(f"   Existing shots_needing_edit from previous iteration: {existing_shots_needing_edit}")
        logger.info(f"   Existing shots_needing_regeneration from previous iteration: {existing_shots_needing_regeneration}")
        
        # Get generated images for S3 URL lookup
        generated_images = state.get("generated_images", {}).get("generated_images", [])
        
        for review in summary.reviews:
            shot_id = review.shot_id
            
            if review.decision == "APPROVE":
                shots_approved.append(shot_id)
                
                # Find the S3 URL for this shot from generated_images
                shot_s3_url = None
                for img in generated_images:
                    if img.get('shot_id') == shot_id:
                        shot_s3_url = img.get('s3_url', '')
                        break
                
                # Save v0 for approved shots (if not already saved)
                if mongodb_client and shot_s3_url:
                    # Check if this is the first approval (no iterations yet)
                    is_first_approval = (
                        edit_loop_iterations.get(shot_id, 0) == 0 and
                        regenerate_loop_iterations.get(shot_id, 0) == 0
                    )
                    
                    if is_first_approval:
                        mongodb_client.update_shot_edited_image(
                            show_id=show_id,
                            episode_number=episode_number,
                            shot_id=shot_id,
                            version="v0",
                            edited_image_s3_url=shot_s3_url,
                            edit_instructions="Approved on first generation",
                            edit_prompt="",
                            edit_timestamp=datetime.now().isoformat()
                        )
                        logger.info(f"  ✓ Saved v0 (approved image) for {shot_id} to MongoDB")
                    
                    # Update MongoDB with approval status
                    mongodb_client.update_shot_approval_status(
                        show_id=show_id,
                        episode_number=episode_number,
                        shot_id=shot_id,
                        approval_status="approved",
                        approval_timestamp=datetime.now().isoformat()
                    )
                    
            elif review.decision == "EDIT":
                # EDIT goes to Agent 7 (existing loop)
                current_iteration = edit_loop_iterations.get(shot_id, 0)
                
                # Only allow up to 3 edit attempts (v1, v2, v3)
                if current_iteration < 3:
                    new_shots_needing_edit.append(shot_id)
                    
                    # Store edit instructions for this shot (so Agent 7 can find them later)
                    shots_edit_instructions[shot_id] = {
                        'edit_instructions': review.edit_instructions or "",
                        'issues_found': review.issues_found if hasattr(review, 'issues_found') else [],
                        'decision': review.decision,
                        'shot_id': shot_id
                    }
                    
                    # On first rejection (iteration 0), save v0 to MongoDB
                    if current_iteration == 0 and mongodb_client:
                        # Find the S3 URL for this shot from generated_images
                        shot_s3_url = None
                        for img in generated_images:
                            if img.get('shot_id') == shot_id:
                                shot_s3_url = img.get('s3_url', '')
                                break
                        
                        if shot_s3_url:
                            mongodb_client.update_shot_edited_image(
                                show_id=show_id,
                                episode_number=episode_number,
                                shot_id=shot_id,
                                version="v0",
                                edited_image_s3_url=shot_s3_url,
                                edit_instructions=review.edit_instructions or "",
                                edit_prompt="",
                                edit_timestamp=datetime.now().isoformat()
                            )
                            logger.info(f"  ✓ Saved v0 (original generated image) for {shot_id} to MongoDB")
                else:
                    # Max retries exceeded
                    logger.warning(f"Shot {shot_id} has exceeded max edit attempts (3)")
                    
                    if mongodb_client:
                        mongodb_client.update_shot_approval_status(
                            show_id=show_id,
                            episode_number=episode_number,
                            shot_id=shot_id,
                            approval_status="max_retries_exceeded",
                            approval_timestamp=datetime.now().isoformat()
                        )
                        
            elif review.decision == "REGENERATE":
                # REGENERATE goes to Agent 15A (new loop)
                current_regenerate_iteration = regenerate_loop_iterations.get(shot_id, 0)
                
                # Only allow up to 1 regeneration attempt to save time (changed from 3)
                if current_regenerate_iteration < 3:
                    new_shots_needing_regeneration.append(shot_id)
                    
                    # On first rejection (iteration 0), save v0 to MongoDB
                    if current_regenerate_iteration == 0 and mongodb_client:
                        # Find the S3 URL for this shot from generated_images
                        shot_s3_url = None
                        for img in generated_images:
                            if img.get('shot_id') == shot_id:
                                shot_s3_url = img.get('s3_url', '')
                                break
                        
                        if shot_s3_url:
                            mongodb_client.update_shot_edited_image(
                                show_id=show_id,
                                episode_number=episode_number,
                                shot_id=shot_id,
                                version="v0",
                                edited_image_s3_url=shot_s3_url,
                                edit_instructions=review.edit_instructions or "Image requires regeneration",
                                edit_prompt="",
                                edit_timestamp=datetime.now().isoformat()
                            )
                            logger.info(f"  ✓ Saved v0 (original image before regeneration) for {shot_id} to MongoDB")
                    
                    logger.info(f"  → Shot {shot_id} requires regeneration (attempt {current_regenerate_iteration + 1}/3) (routing to Agent 15A)")
                else:
                    # Max regeneration attempts exceeded
                    logger.warning(f"Shot {shot_id} has exceeded max regeneration attempts (3)")
                    
                    if mongodb_client:
                        mongodb_client.update_shot_approval_status(
                            show_id=show_id,
                            episode_number=episode_number,
                            shot_id=shot_id,
                            approval_status="max_retries_exceeded",
                            approval_timestamp=datetime.now().isoformat()
                        )
        
        # Count shots by decision type for logging
        edit_count = sum(1 for r in summary.reviews if r.decision == "EDIT")
        regenerate_count = sum(1 for r in summary.reviews if r.decision == "REGENERATE")
        
        # Get list of shot_ids that were just reviewed
        reviewed_shot_ids = [r.shot_id for r in summary.reviews]
        
        # Merge with existing lists:
        # 1. Remove shots that were just reviewed from existing lists (they have new decisions now)
        # 2. Add new decisions
        # 3. Remove duplicates
        
        # For shots_needing_edit: keep existing ones not just reviewed + add new ones
        shots_needing_edit = [sid for sid in existing_shots_needing_edit if sid not in reviewed_shot_ids]
        shots_needing_edit.extend(new_shots_needing_edit)
        shots_needing_edit = list(set(shots_needing_edit))  # Remove duplicates
        
        # For shots_needing_regeneration: keep existing ones not just reviewed + add new ones
        shots_needing_regeneration = [sid for sid in existing_shots_needing_regeneration if sid not in reviewed_shot_ids]
        shots_needing_regeneration.extend(new_shots_needing_regeneration)
        shots_needing_regeneration = list(set(shots_needing_regeneration))  # Remove duplicates
        
        # Clean up edit instructions for approved shots or shots no longer in edit list
        for shot_id in list(shots_edit_instructions.keys()):
            if shot_id not in shots_needing_edit:
                # Remove if shot is no longer in edit list
                del shots_edit_instructions[shot_id]
        
        logger.info(f"\nAgent 15 Review Summary (This Review):")
        logger.info(f"   ✅ Approved: {len(shots_approved)}")
        logger.info(f"   ✏️  Need Edit: {edit_count} → Agent 7")
        logger.info(f"   🔄 Need Regeneration: {regenerate_count} → Agent 15A")
        
        logger.info(f"\nCumulative Status (After Merge):")
        logger.info(f"   ✏️  Total shots needing edit: {len(shots_needing_edit)} → {shots_needing_edit}")
        logger.info(f"   🔄 Total shots needing regeneration: {len(shots_needing_regeneration)} → {shots_needing_regeneration}")
        logger.info(f"   ✅ Total approved so far: {len(state.get('shots_approved', []) + shots_approved)}")
        
        # Determine next step based on decisions (prioritize regeneration over edit)
        if len(shots_needing_regeneration) > 0:
            next_agent = "agent_15A"
            pipeline_status = "running"
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 Starting agent_15-15A-14-15 REGENERATION LOOP for {len(shots_needing_regeneration)} shot(s):")
            for shot_id in shots_needing_regeneration:
                logger.info(f"    - {shot_id}")
            logger.info(f"{'='*80}")
        elif len(shots_needing_edit) > 0:
            next_agent = "agent_7"
            pipeline_status = "running"
            logger.info(f"\n{'='*80}")
            logger.info(f"✏️  Starting agent_15-7-15 EDIT LOOP for {len(shots_needing_edit)} shot(s):")
            for shot_id in shots_needing_edit:
                logger.info(f"    - {shot_id}")
            logger.info(f"{'='*80}")
        else:
            next_agent = "final_approval_checkpoint"
            pipeline_status = "waiting_for_final_approval"
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ All shots processed. Routing to Final Approval Checkpoint")
            logger.info(f"{'='*80}")

        # Update state
        return {
            **state,
            "image_reviews": {
                "summary": {
                    "total_images": summary.total_images,
                    "approved": summary.approved,
                    "edit": summary.edit,
                    "regenerate": summary.regenerate,
                    "critical_issues": summary.critical_issues
                },
                "reviews": [asdict(review) for review in summary.reviews]
            },
            "agent15_status": "completed",
            "current_agent": next_agent,
            "pipeline_status": pipeline_status,
            "output_files": state.get("output_files", []) + [report_path],
            "shots_needing_edit": shots_needing_edit,
            "shots_needing_regeneration": shots_needing_regeneration,
            "shots_approved": state.get("shots_approved", []) + shots_approved,
            "edit_loop_iterations": edit_loop_iterations,
            "regenerate_loop_iterations": regenerate_loop_iterations,
            "shots_edit_instructions": shots_edit_instructions,
        }

    except Exception as e:
        logger.error(f"Agent 15 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent15_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_7_shot_editor_node(state: Phase2State) -> Phase2State:
    """
    Agent 7: Shot Editor Node
    Edits shots that need improvements based on Agent 15 feedback
    """
    logger.info("\n" + "="*60)
    logger.info("✏AGENT 7: SHOT EDITOR")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)
    
    try:
        # Get shots needing edit
        shots_needing_edit = state.get("shots_needing_edit", [])
        
        if not shots_needing_edit:
            logger.warning("No shots need editing")
            return {
                **state,
                "agent7_status": "completed",
                "current_agent": "final_approval_checkpoint",
                "pipeline_status": "waiting_for_final_approval",
            }
        
        # Initialize agent
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ShotEditorAgent(gemini_api_key=api_key)
        
        # Get stored edit instructions collected by Agent 15 across iterations
        shots_edit_instructions = state.get("shots_edit_instructions", {})
        
        # Current review payload (used as fallback if instructions missing)
        image_reviews = state.get("image_reviews", {})
        reviews = image_reviews.get("reviews", [])
        
        # Get original prompts from modified_prompts (Agent 13)
        modified_prompts_data = state.get("modified_prompts", {})
        modified_prompts = modified_prompts_data.get("modified_shots", [])
        
        # Get current edit loop iterations
        edit_loop_iterations = state.get("edit_loop_iterations", {})
        
        # Prepare shots data for editing
        shots_to_edit = []
        for shot_id in shots_needing_edit:
            # SAFEGUARD: Check if shot has already reached max iterations (3)
            current_iteration = edit_loop_iterations.get(shot_id, 0)
            if current_iteration >= 3:
                logger.warning(f"Skipping {shot_id} - already at max edit iterations (3). Current iteration: {current_iteration}")
                continue
            
            # Prefer persisted instructions from previous review loops
            review = shots_edit_instructions.get(shot_id)
            
            if not review:
                review = next((r for r in reviews if r.get('shot_id') == shot_id), None)
                if review:
                    # cache for subsequent iterations
                    shots_edit_instructions[shot_id] = review
            
            mongo_fallback_instructions = ""
            if not review:
                logger.warning(f"No cached review found for shot {shot_id}. Checking MongoDB for instructions.")
                mongodb_client = state.get("mongodb_client")
                if mongodb_client:
                    try:
                        query = {"show_id": state["show_id"], "annotated_shots": {"$exists": True}}
                        raw_doc = mongodb_client.shots_collection.find_one(query)
                        if raw_doc and "annotated_shots" in raw_doc:
                            for shot in raw_doc["annotated_shots"]:
                                if shot.get("shot_id") == shot_id:
                                    edited_image_s3 = shot.get("edited_image_s3") or {}
                                    if "v0" in edited_image_s3:
                                        mongo_fallback_instructions = edited_image_s3["v0"].get("edit_instructions", "")
                                    break
                    except Exception as e:
                        logger.warning(f"  Error fetching edit instructions for {shot_id} from MongoDB: {e}")
                if mongo_fallback_instructions:
                    review = {
                        "shot_id": shot_id,
                        "edit_instructions": mongo_fallback_instructions,
                        "issues_found": [],
                        "decision": "EDIT"
                    }
                else:
                    logger.warning(f"No edit instructions available for {shot_id}. Skipping.")
                    continue
            
            # Find original prompt
            prompt_data = next((p for p in modified_prompts if p['shot_id'] == shot_id), None)
            original_prompt = prompt_data.get('corrected_prompt', '') if prompt_data else ''
            
            # Get current S3 URL from MongoDB (latest version)
            current_iteration = edit_loop_iterations.get(shot_id, 0)
            current_s3_url = None
            mongo_edit_instructions = ""
            
            # Try to get the latest edited image directly from MongoDB
            mongodb_client = state.get("mongodb_client")
            if mongodb_client:
                try:
                    query = {"show_id": state["show_id"], "annotated_shots": {"$exists": True}}
                    raw_doc = mongodb_client.shots_collection.find_one(query)
                    
                    if raw_doc and "annotated_shots" in raw_doc:
                        for shot in raw_doc["annotated_shots"]:
                            if shot.get("shot_id") == shot_id:
                                edited_image_s3 = shot.get("edited_image_s3")
                                if edited_image_s3 and isinstance(edited_image_s3, dict):
                                    versions = list(edited_image_s3.keys())
                                    if versions:
                                        versions.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0, reverse=True)
                                        latest_version = versions[0]
                                        latest_image_data = edited_image_s3[latest_version]
                                        current_s3_url = latest_image_data.get('s3_url', '')
                                        mongo_edit_instructions = latest_image_data.get('edit_instructions', '')
                                        logger.info(f"  ✓ Found {shot_id} ({latest_version}) from MongoDB: {current_s3_url[:80]}...")
                                    else:
                                        logger.warning(f"  No edited images found for {shot_id} in MongoDB")
                                else:
                                    logger.warning(f"  No edited_image_s3 field found for {shot_id} in MongoDB")
                                break
                except Exception as e:
                    logger.warning(f"  Error fetching latest edited image from MongoDB: {e}")
            
            # Fallback to state data if MongoDB fails
            if not current_s3_url:
                if current_iteration == 0:
                    # Get from generated_images
                    generated_images = state.get("generated_images", {}).get("generated_images", [])
                    for img in generated_images:
                        if img.get('shot_id') == shot_id:
                            current_s3_url = img.get('s3_url', '')
                            break
                else:
                    # Get from previous edit (edited_shots)
                    edited_shots = state.get("edited_shots", {}).get("edit_results", [])
                    for edit in edited_shots:
                        if edit.get('shot_id') == shot_id:
                            current_s3_url = edit.get('edited_s3_url', '')
                            break
            
            if not current_s3_url:
                logger.warning(f"No S3 URL found for shot {shot_id}")
                continue
            
            shots_to_edit.append({
                'shot_id': shot_id,
                'current_s3_url': current_s3_url,
                'edit_instructions': mongo_edit_instructions or review.get('edit_instructions', ''),
                'issues_found': review.get('issues_found', []),
                'original_prompt': original_prompt
            })
        
        # Build product shot context
        raw_shots_a7 = state.get("shot_list_request", {}).get("shots", [])
        product_shot_ids_a7 = {s.get("shot_id") for s in raw_shots_a7 if s.get("product_present")} if raw_shots_a7 else set()
        if product_shot_ids_a7:
            logger.info(f"Agent 7: product shots detected: {product_shot_ids_a7}")

        # Edit shots
        edit_results = agent.edit_shots_batch(
            shots_to_edit=shots_to_edit,
            edit_loop_iterations=edit_loop_iterations,
            show_id=state["show_id"],
            product_shot_ids=product_shot_ids_a7
        )
        
        # Save edit report
        report_path = agent.save_edit_report()
        
        # Update edit loop iterations
        new_edit_loop_iterations = {**edit_loop_iterations}
        for shot_id in shots_needing_edit:
            new_edit_loop_iterations[shot_id] = new_edit_loop_iterations.get(shot_id, 0) + 1
        
        # Save edited images to MongoDB (using show_id for lookup)
        mongodb_client = state.get("mongodb_client")
        episode_number = state.get("episode_number", 0)
        
        if mongodb_client:
            for edit_result in edit_results.get("edit_results", []):
                if edit_result.get("success"):
                    success = mongodb_client.update_shot_edited_image(
                        show_id=state["show_id"],
                        episode_number=episode_number,
                        shot_id=edit_result["shot_id"],
                        version=edit_result["edit_version"],
                        edited_image_s3_url=edit_result["edited_s3_url"],
                        edit_instructions=edit_result["edit_instructions"],
                        edit_prompt=edit_result["edit_prompt"],
                        edit_timestamp=edit_result["edit_timestamp"]
                    )
                    if success:
                        logger.info(f"  ✓ Saved {edit_result['shot_id']} ({edit_result['edit_version']}) to MongoDB")
                    else:
                        logger.warning(f"  Failed to save {edit_result['shot_id']} ({edit_result['edit_version']}) to MongoDB")
        
        logger.info(f"\nAgent 7 completed: Edited {edit_results.get('total_edited', 0)} shots")
        
        # Merge new edits with existing edited_shots — do NOT overwrite.
        # Overwriting would erase URLs from previous edit iterations, making
        # _get_latest_image_url blind to earlier versions.
        existing_edited = state.get("edited_shots", {})
        existing_edit_results = existing_edited.get("edit_results", [])
        new_edit_results = edit_results.get("edit_results", [])
        new_shot_ids = {r["shot_id"] for r in new_edit_results if r.get("shot_id")}
        merged_edit_results = [r for r in existing_edit_results if r.get("shot_id") not in new_shot_ids] + new_edit_results
        merged_edited_shots = {**existing_edited, "edit_results": merged_edit_results}

        # Route back to Agent 15 for re-review
        return {
            **state,
            "edited_shots": merged_edited_shots,
            "agent7_status": "completed",
            "current_agent": "agent_15_review",
            "pipeline_status": "running",
            "output_files": state.get("output_files", []) + [report_path],
            "edit_loop_iterations": new_edit_loop_iterations,
        }
        
    except Exception as e:
        logger.error(f"Agent 7 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "agent7_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_15_review_loop_node(state: Phase2State) -> Phase2State:
    """
    Agent 15: Image Reviewer Node (For Edit Loop Re-review)
    Reviews edited images from Agent 7
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 15: IMAGE REVIEWER (RE-REVIEW)")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("="*60)
    
    try:
        from dataclasses import asdict
        # Initialize agent
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ImageReviewAgent(api_key=api_key)
        
        # Get shots that need re-review from state
        shots_needing_edit = state.get("shots_needing_edit", [])
        
        if not shots_needing_edit:
            logger.warning("No shots need re-review")
            return {
                **state,
                "current_agent": "final_approval_checkpoint",
                "pipeline_status": "waiting_for_final_approval",
            }
        
        # Get shot designs from Agent 12 and modified prompts from Agent 13
        shot_designs_data = state.get("shot_designs", {})
        shot_designs = shot_designs_data.get("designs", [])
        
        modified_prompts_data = state.get("modified_prompts", {})
        modified_prompts = modified_prompts_data.get("modified_shots", [])
        
        # Fetch latest edited images from MongoDB for each shot that needs re-review
        mongodb_client = state.get("mongodb_client")
        edited_images_for_review = []
        
        if mongodb_client:
            logger.info(f"Fetching latest edited images from MongoDB for {len(shots_needing_edit)} shots...")
            
            # Get the shots document from MongoDB using raw query (not Pydantic model)
            try:
                # Query MongoDB directly to get raw document
                query = {"show_id": state["show_id"], "annotated_shots": {"$exists": True}}
                raw_doc = mongodb_client.shots_collection.find_one(query)
                
                if raw_doc and "annotated_shots" in raw_doc:
                    logger.info(f"  ✓ Found document in MongoDB with {len(raw_doc['annotated_shots'])} shots")
                    
                    for shot in raw_doc["annotated_shots"]:
                        shot_id = shot.get("shot_id", "")
                        if shot_id in shots_needing_edit:
                            # Get the latest edited image version (v1, v2, v3, etc.)
                            edited_image_s3 = shot.get("edited_image_s3")
                            
                            if edited_image_s3 and isinstance(edited_image_s3, dict):
                                # Find the highest version number
                                versions = list(edited_image_s3.keys())
                                if versions:
                                    # Sort versions to get the latest (v3 > v2 > v1 > v0)
                                    versions.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0, reverse=True)
                                    latest_version = versions[0]
                                    latest_image_data = edited_image_s3[latest_version]
                                    
                                    s3_url = latest_image_data.get('s3_url', '')
                                    if s3_url:
                                        edited_images_for_review.append({
                                            'shot_id': shot_id,
                                            's3_url': s3_url,
                                            'local_path': '',  # Not needed for review
                                            'version': latest_version
                                        })
                                        logger.info(f"  ✓ Found {shot_id} ({latest_version}): {s3_url[:50]}...")
                                    else:
                                        logger.warning(f"  No S3 URL in {shot_id} ({latest_version})")
                                else:
                                    logger.warning(f"  No versions found in edited_image_s3 for {shot_id}")
                            else:
                                logger.warning(f"  No edited_image_s3 field found for {shot_id}")
                else:
                    logger.warning("No shots document found in MongoDB")
            except Exception as e:
                logger.warning(f"Error fetching from MongoDB: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("No MongoDB client available, falling back to state data")
            # Fallback to state data if MongoDB is not available
            edited_shots_data = state.get("edited_shots", {})
            edited_results = edited_shots_data.get("edit_results", [])
            for edit_result in edited_results:
                if edit_result.get("success") and edit_result.get("shot_id") in shots_needing_edit:
                    edited_images_for_review.append({
                        'shot_id': edit_result['shot_id'],
                        's3_url': edit_result['edited_s3_url'],
                        'local_path': '',  # Not needed for review
                        'version': edit_result.get('edit_version', 'unknown')
                    })
        
        # Review edited images
        summary = agent.review_all_images(
            generated_images=edited_images_for_review,
            shot_designs=shot_designs,
            modified_prompts=modified_prompts
        )
        
        # Save review report
        report_path = agent.save_review_report(summary)
        
        logger.info(f"\nAgent 15 re-review completed: {summary.total_images} images")
        logger.info(f"   - Approved: {summary.approved}")
        logger.info(f"   - Edit required: {summary.edit}")
        logger.info(f"   - Regenerate required: {summary.regenerate}")
        
        # Process reviews and update state
        # Start with existing lists from state (preserve across iterations)
        existing_shots_needing_edit = state.get("shots_needing_edit", [])
        existing_shots_needing_regeneration = state.get("shots_needing_regeneration", [])
        
        # Get existing edit instructions dict
        shots_edit_instructions = state.get("shots_edit_instructions", {})
        
        # Create new lists for this review iteration
        new_shots_needing_edit = []
        new_shots_needing_regeneration = []
        shots_approved_in_loop = []
        shots_max_retries = state.get("shots_max_retries", [])
        edit_loop_iterations = state.get("edit_loop_iterations", {})
        regenerate_loop_iterations = state.get("regenerate_loop_iterations", {})
        
        # Get the shots that were actually edited by Agent 7 (from state)
        edited_shots_data = state.get("edited_shots", {})
        edited_results = edited_shots_data.get("edit_results", [])
        edited_shot_ids = [result["shot_id"] for result in edited_results if result.get("success")]
        
        mongodb_client = state.get("mongodb_client")
        episode_number = state.get("episode_number", 0)
        
        for review in summary.reviews:
            shot_id = review.shot_id
            
            # Only process reviews for shots that were actually edited by Agent 7
            if shot_id not in edited_shot_ids:
                logger.info(f"Skipping review for {shot_id} (not edited by Agent 7)")
                continue
            
            if review.decision == "APPROVE":
                shots_approved_in_loop.append(shot_id)
                
                # Update MongoDB with approval status
                if mongodb_client:
                    mongodb_client.update_shot_approval_status(
                        show_id=state["show_id"],
                        episode_number=episode_number,
                        shot_id=shot_id,
                        approval_status="approved",
                        approval_timestamp=datetime.now().isoformat()
                    )
                    
            elif review.decision == "EDIT":
                # EDIT goes to Agent 7 (existing loop)
                current_iteration = edit_loop_iterations.get(shot_id, 0)
                
                logger.info(f"Shot {shot_id}: decision=EDIT, current_iteration={current_iteration}")
                
                # Check if we've reached max retries (3 edits = v1, v2, v3)
                if current_iteration < 3:
                    new_shots_needing_edit.append(shot_id)
                    
                    # Store edit instructions for this shot
                    shots_edit_instructions[shot_id] = {
                        'edit_instructions': review.edit_instructions or "",
                        'issues_found': review.issues_found if hasattr(review, 'issues_found') else [],
                        'decision': review.decision,
                        'shot_id': shot_id
                    }
                    
                    logger.info(f"   → Adding to shots_needing_edit (iteration {current_iteration} < 3)")
                else:
                    # Max retries exceeded
                    logger.warning(f"Shot {shot_id} has exceeded max edit attempts (3)")
                    shots_max_retries.append(shot_id)
                    
                    if mongodb_client:
                        mongodb_client.update_shot_approval_status(
                            show_id=state["show_id"],
                            episode_number=episode_number,
                            shot_id=shot_id,
                            approval_status="max_retries_exceeded",
                            approval_timestamp=datetime.now().isoformat()
                        )
                        
            elif review.decision == "REGENERATE":
                # REGENERATE goes to Agent 15A (new loop)
                current_regenerate_iteration = regenerate_loop_iterations.get(shot_id, 0)
                
                logger.info(f"Shot {shot_id}: decision=REGENERATE, current_regenerate_iteration={current_regenerate_iteration}")
                
                # Check if we've reached max retries (3 regenerations = v1, v2, v3)
                if current_regenerate_iteration < 3:
                    new_shots_needing_regeneration.append(shot_id)
                    logger.info(f"   → Adding to shots_needing_regeneration (iteration {current_regenerate_iteration} < 3)")
                else:
                    # Max regeneration attempts exceeded
                    logger.warning(f"Shot {shot_id} has exceeded max regeneration attempts (3)")
                    shots_max_retries.append(shot_id)
                    
                    if mongodb_client:
                        mongodb_client.update_shot_approval_status(
                            show_id=state["show_id"],
                            episode_number=episode_number,
                            shot_id=shot_id,
                            approval_status="max_retries_exceeded",
                            approval_timestamp=datetime.now().isoformat()
                        )
        
        # Get list of shot_ids that were just reviewed
        reviewed_shot_ids = [r.shot_id for r in summary.reviews]
        
        # Merge with existing lists:
        # For shots_needing_edit: keep existing ones not just reviewed + add new ones
        shots_needing_edit = [sid for sid in existing_shots_needing_edit if sid not in reviewed_shot_ids]
        shots_needing_edit.extend(new_shots_needing_edit)
        shots_needing_edit = list(set(shots_needing_edit))  # Remove duplicates
        
        # For shots_needing_regeneration: keep existing ones not just reviewed + add new ones
        shots_needing_regeneration = [sid for sid in existing_shots_needing_regeneration if sid not in reviewed_shot_ids]
        shots_needing_regeneration.extend(new_shots_needing_regeneration)
        shots_needing_regeneration = list(set(shots_needing_regeneration))  # Remove duplicates
        
        # Clean up edit instructions for approved shots or shots no longer in edit list
        for shot_id in list(shots_edit_instructions.keys()):
            if shot_id not in shots_needing_edit:
                del shots_edit_instructions[shot_id]
        
        # Count shots by decision type for logging
        edit_count = sum(1 for r in summary.reviews if r.decision == "EDIT")
        regenerate_count = sum(1 for r in summary.reviews if r.decision == "REGENERATE")
        
        logger.info(f"\nRe-review Summary:")
        logger.info(f"   Approved: {len(shots_approved_in_loop)}")
        logger.info(f"   ✏️  Need Edit: {edit_count} → Agent 7")
        logger.info(f"   🔄 Need Regeneration: {regenerate_count} → Agent 15A")
        logger.info(f"   Max Retries Exceeded: {len(shots_max_retries)}")
        
        logger.info(f"\nCumulative Status (After Merge):")
        logger.info(f"   ✏️  Total shots needing edit: {len(shots_needing_edit)} → {shots_needing_edit}")
        logger.info(f"   🔄 Total shots needing regeneration: {len(shots_needing_regeneration)} → {shots_needing_regeneration}")
        logger.info(f"   ✅ Total approved so far: {len(state.get('shots_approved', []) + shots_approved_in_loop)}")
        
        # Determine next step (prioritize regeneration over edit)
        if len(shots_needing_regeneration) > 0:
            next_agent = "agent_15A"
            pipeline_status = "running"
            logger.info(f"\n→ Routing to Agent 15A for {len(shots_needing_regeneration)} regenerations")
        elif len(shots_needing_edit) > 0:
            next_agent = "agent_7"
            pipeline_status = "running"
            logger.info(f"\n→ Routing back to Agent 7 for {len(shots_needing_edit)} more edits")
        else:
            next_agent = "final_approval_checkpoint"
            pipeline_status = "waiting_for_final_approval"
            logger.info(f"\n→ Edit loop complete. Routing to Final Approval Checkpoint")
        
        # Update state
        return {
            **state,
            "image_reviews": {
                "summary": {
                    "total_images": summary.total_images,
                    "approved": summary.approved,
                    "edit": summary.edit,
                    "regenerate": summary.regenerate,
                    "critical_issues": summary.critical_issues
                },
                "reviews": [asdict(review) for review in summary.reviews]
            },
            "current_agent": next_agent,
            "pipeline_status": pipeline_status,
            "output_files": state.get("output_files", []) + [report_path],
            "shots_needing_edit": shots_needing_edit,
            "shots_needing_regeneration": shots_needing_regeneration,
            "shots_approved": state.get("shots_approved", []) + shots_approved_in_loop,
            "shots_max_retries": shots_max_retries,
            "edit_loop_iterations": edit_loop_iterations,
            "regenerate_loop_iterations": regenerate_loop_iterations,
            "shots_edit_instructions": shots_edit_instructions,
        }
        
    except Exception as e:
        logger.error(f"Agent 15 re-review failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def _get_latest_image_url(shot_id: str, state: Phase2State) -> Optional[str]:
    """
    Resolve the latest image URL for a shot across all output stages.
    Priority: product_corrected_images → edited_shots → generated_images
    """
    corrected = state.get("product_corrected_images", {})
    if corrected.get(shot_id):
        return corrected[shot_id]

    edited_shots = state.get("edited_shots", {})
    edit_results = edited_shots.get("edit_results", [])
    for edit in edit_results:
        if edit.get("shot_id") == shot_id and edit.get("edited_s3_url"):
            return edit["edited_s3_url"]

    generated_images = state.get("generated_images", {})
    gen_list = generated_images.get("generated_images", [])
    for img in gen_list:
        if img.get("shot_id") == shot_id and img.get("s3_url"):
            return img["s3_url"]

    return None


def agent_16_product_reviewer_node(state: Phase2State) -> Phase2State:
    """
    Agent 16: Product Fidelity Reviewer Node
    Reviews product shots to verify shape, size, and text match the reference product.
    """
    logger.info("\n" + "=" * 60)
    logger.info("AGENT 16: PRODUCT FIDELITY REVIEWER")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("=" * 60)

    try:
        product_image_url = state.get("product_image_url")
        if not product_image_url:
            product_shot_ids = _get_product_shot_ids(state)
            _log_product_review_skip(product_shot_ids, product_image_url)
            return {
                **state,
                "current_agent": "final_approval_checkpoint",
                "pipeline_status": "waiting_for_final_approval",
                "shots_product_approved": state.get("shots_approved", []),
                "shots_needing_product_fix": [],
            }

        product_shot_ids = _get_product_shot_ids(state)

        if not product_shot_ids:
            _log_product_review_skip(product_shot_ids, product_image_url)
            return {
                **state,
                "current_agent": "final_approval_checkpoint",
                "pipeline_status": "waiting_for_final_approval",
                "shots_product_approved": state.get("shots_approved", []),
                "shots_needing_product_fix": [],
            }

        # Determine which shots to review in this run.
        # On the first pass: review all shots_approved by Agent 15.
        # On subsequent passes (after Agent 18 edits): review shots that were in shots_needing_product_fix.
        existing_needing_fix = state.get("shots_needing_product_fix", [])
        shots_approved_by_15 = state.get("shots_approved", [])

        if existing_needing_fix:
            # Re-review shots that just went through Agents 17+18
            shots_to_review = [s for s in existing_needing_fix if s in product_shot_ids]
        else:
            # First pass: review all product shots that Agent 15 approved
            shots_to_review = [s for s in shots_approved_by_15 if s in product_shot_ids]

        if not shots_to_review:
            logger.info("No product shots to review — routing to final_checkpoint")
            return {
                **state,
                "current_agent": "final_approval_checkpoint",
                "pipeline_status": "waiting_for_final_approval",
                "shots_needing_product_fix": [],
            }

        # Build latest image URL map for the shots under review
        generated_images_map = {sid: _get_latest_image_url(sid, state) for sid in shots_to_review}
        generated_images_map = {k: v for k, v in generated_images_map.items() if v}

        product_review_iterations = state.get("product_review_iterations", {})

        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ProductReviewerAgent(api_key=api_key)

        results = agent.review_product_shots(
            product_shot_ids=product_shot_ids,
            shots_to_review=shots_to_review,
            generated_images=generated_images_map,
            product_image_url=product_image_url,
            product_review_iterations=product_review_iterations,
        )

        # Merge review_results into state
        merged_review_results = {**state.get("product_review_results", {}), **results["review_results"]}

        # Update iteration counts for shots that failed (not force-passed)
        new_iterations = {**product_review_iterations}
        for shot_id in results["shots_failing"]:
            new_iterations[shot_id] = new_iterations.get(shot_id, 0) + 1

        # Accumulate approved shots (passing + force-passed)
        already_approved = state.get("shots_product_approved", [])
        newly_approved = results["shots_passing"] + results["shots_force_passed"]
        all_product_approved = list(set(already_approved + newly_approved))

        shots_needing_fix = results["shots_failing"]

        if shots_needing_fix:
            next_agent = "product_prompt_gen"
            pipeline_status = "running"
            logger.info(f"→ {len(shots_needing_fix)} shots need product fix: {shots_needing_fix}")
        else:
            next_agent = "final_approval_checkpoint"
            pipeline_status = "waiting_for_final_approval"
            logger.info("→ All product shots passed — routing to final_checkpoint")

        return {
            **state,
            "product_review_results": merged_review_results,
            "product_review_iterations": new_iterations,
            "shots_needing_product_fix": shots_needing_fix,
            "shots_product_approved": all_product_approved,
            "current_agent": next_agent,
            "pipeline_status": pipeline_status,
        }

    except Exception as e:
        logger.error(f"Agent 16 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_17_product_prompt_gen_node(state: Phase2State) -> Phase2State:
    """
    Agent 17: Product Position & Replacement Prompt Generator Node
    Generates Nano Banana prompts describing how to replace incorrect products.
    """
    logger.info("\n" + "=" * 60)
    logger.info("AGENT 17: PRODUCT REPLACEMENT PROMPT GENERATOR")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("=" * 60)

    try:
        shots_needing_fix = state.get("shots_needing_product_fix", [])
        product_image_url = state.get("product_image_url", "")

        if not shots_needing_fix:
            logger.warning("No shots need product fix — skipping Agent 17")
            return {
                **state,
                "current_agent": "product_editor",
                "pipeline_status": "running",
            }

        # Build latest image URL map
        generated_images_map = {sid: _get_latest_image_url(sid, state) for sid in shots_needing_fix}
        generated_images_map = {k: v for k, v in generated_images_map.items() if v}

        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ProductPromptGenAgent(api_key=api_key)

        new_prompts = agent.generate_prompts_batch(
            shots_needing_fix=shots_needing_fix,
            generated_images=generated_images_map,
            product_image_url=product_image_url,
            product_review_results=state.get("product_review_results", {}),
        )

        merged_prompts = {**state.get("product_fix_prompts", {}), **new_prompts}

        return {
            **state,
            "product_fix_prompts": merged_prompts,
            "current_agent": "product_editor",
            "pipeline_status": "running",
        }

    except Exception as e:
        logger.error(f"Agent 17 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_18_product_editor_node(state: Phase2State) -> Phase2State:
    """
    Agent 18: Product Image Editor Node
    Uses Nano Banana to replace incorrect products using prompts from Agent 17.
    """
    logger.info("\n" + "=" * 60)
    logger.info("AGENT 18: PRODUCT EDITOR (NANO BANANA)")
    job_id = state.get("job_id")
    if job_id:
        logger.info(f"Job ID: {job_id}")
    logger.info("=" * 60)

    try:
        shots_needing_fix = state.get("shots_needing_product_fix", [])
        product_image_url = state.get("product_image_url", "")
        product_fix_prompts = state.get("product_fix_prompts", {})
        product_review_iterations = state.get("product_review_iterations", {})

        if not shots_needing_fix:
            logger.warning("No shots need product editing — skipping Agent 18")
            return {
                **state,
                "current_agent": "product_reviewer",
                "pipeline_status": "running",
            }

        # Build latest image URL map (before editing)
        generated_images_map = {sid: _get_latest_image_url(sid, state) for sid in shots_needing_fix}
        generated_images_map = {k: v for k, v in generated_images_map.items() if v}

        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ProductEditorAgent(api_key=api_key)

        edit_results = agent.edit_shots_batch(
            shots_needing_fix=shots_needing_fix,
            generated_images=generated_images_map,
            product_image_url=product_image_url,
            product_fix_prompts=product_fix_prompts,
            product_review_iterations=product_review_iterations,
            show_id=state.get("show_id", ""),
        )

        # Merge corrected images into state
        merged_corrected = {**state.get("product_corrected_images", {}), **edit_results["corrected_images"]}

        # Save corrected images to MongoDB if client is available
        mongodb_client = state.get("mongodb_client")
        episode_number = state.get("episode_number", 0)
        show_id = state.get("show_id", "")

        if mongodb_client and edit_results["corrected_images"]:
            for shot_id, corrected_url in edit_results["corrected_images"].items():
                iteration = product_review_iterations.get(shot_id, 0) + 1
                try:
                    mongodb_client.update_shot_edited_image(
                        show_id=show_id,
                        episode_number=episode_number,
                        shot_id=shot_id,
                        version=f"product_fix_v{iteration}",
                        edited_image_s3_url=corrected_url,
                        edit_instructions=product_fix_prompts.get(shot_id, ""),
                        edit_prompt=product_fix_prompts.get(shot_id, ""),
                        edit_timestamp=datetime.now().isoformat(),
                    )
                    logger.info(f"  Saved product_fix_v{iteration} for {shot_id} to MongoDB")
                except Exception as db_err:
                    logger.warning(f"  MongoDB save failed for {shot_id}: {db_err}")

        logger.info(f"Agent 18 complete: corrected {len(edit_results['corrected_images'])} shots")

        return {
            **state,
            "product_corrected_images": merged_corrected,
            "current_agent": "product_reviewer",
            "pipeline_status": "running",
        }

    except Exception as e:
        logger.error(f"Agent 18 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def final_approval_checkpoint(state: Phase2State) -> Phase2State:
    """
    Final Human Approval Checkpoint Node
    Pauses workflow and waits for final human approval after all edits
    """
    logger.info("\n" + "="*60)
    logger.info("🧑 FINAL HUMAN APPROVAL CHECKPOINT")
    logger.info("="*60)
    
    # Check if human has already provided a decision
    if state.get("final_approval_decision") is not None:
        decision = state["final_approval_decision"]
        logger.info(f"✓ Final human decision already received: {decision}")
        
        if decision:
            logger.info("   → Final approval granted - workflow complete")
            return {
                **state,
                "pipeline_status": "completed",
                "current_agent": "completed",
                "requires_human_feedback": False,
                "feedback_agent": None,
            }
        else:
            logger.info("   → Final approval rejected - ending workflow")
            return {
                **state,
                "pipeline_status": "rejected",
                "current_agent": "completed",
                "requires_human_feedback": False,
                "feedback_agent": None,
            }
    
    # No decision yet - pause workflow
    logger.info("⏸Workflow paused. Awaiting final human approval...")
    logger.info("   - All image generation and editing complete")
    logger.info("   - Review final images and approve/reject")
    
    shots_approved = state.get("shots_approved", [])
    shots_max_retries = state.get("shots_max_retries", [])
    
    logger.info(f"\nFinal Status:")
    logger.info(f"   Approved shots: {len(shots_approved)}")
    logger.error(f"   Max retries exceeded: {len(shots_max_retries)}")
    
    return {
        **state,
        "pipeline_status": "waiting_for_final_approval",
        "current_agent": "final_approval_checkpoint",
        "requires_human_feedback": True,
        "feedback_agent": "final_approval",
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue(state: Phase2State) -> Literal["agent_2", "human_checkpoint", "end", "failed"]:
    """
    Determines next step based on current agent status
    """
    current = state.get("current_agent", "")
    pipeline_status = state.get("pipeline_status", "running")

    # Check for failures
    if pipeline_status == "failed":
        return "failed"

    # Check for completion
    if pipeline_status == "completed" or pipeline_status == "rejected":
        return "end"

    # Route to next agent
    if current == "agent_2":
        return "agent_2"
    elif current == "human_checkpoint":
        return "human_checkpoint"
    elif current == "completed":
        return "end"

    return "failed"


def route_after_approval(state: Phase2State) -> Literal["agent_2", "end", "wait"]:
    """
    After Human Approval: Route based on human decision
    """
    pipeline_status = state.get("pipeline_status", "")

    if pipeline_status == "failed":
        return "end"

    # If still waiting for human approval, stay in wait state
    if pipeline_status == "waiting_for_approval":
        return "wait"

    # Check human decision
    current_agent = state.get("current_agent", "")

    if current_agent == "agent_2":
        logger.info("   → Human approved! Routing to Agent 2 (Image Prompt Generator)")
        return "agent_2"
    elif current_agent == "completed":
        logger.info("   → Human rejected. Ending workflow")
        return "end"

    # Default: wait
    return "wait"


# ============================================================================
# CONDITIONAL ROUTING (Additional)
# ============================================================================

def route_after_prompt_approval(state: Phase2State) -> Literal["agent_14", "end", "wait"]:
    """
    After Prompt Approval: Route based on human decision
    """
    pipeline_status = state.get("pipeline_status", "")

    if pipeline_status == "failed":
        return "end"

    # If still waiting for prompt approval, stay in wait state
    if pipeline_status == "waiting_for_prompt_approval":
        return "wait"

    # Check human decision
    current_agent = state.get("current_agent", "")

    if current_agent == "agent_14":
        logger.info("   → Prompts approved! Routing to Agent 14 (Imagen Generator)")
        return "agent_14"
    elif current_agent == "completed":
        logger.info("   → Prompts rejected. Ending workflow")
        return "end"

    # Default: wait
    return "wait"


def route_after_agent_15(state: Phase2State) -> Literal["agent_15A", "agent_7", "product_reviewer", "final_checkpoint", "end"]:
    """
    After Agent 15: Route based on whether shots need regeneration, editing, or product review.
    """
    pipeline_status = state.get("pipeline_status", "")
    shots_needing_regeneration = state.get("shots_needing_regeneration", [])
    shots_needing_edit = state.get("shots_needing_edit", [])

    if pipeline_status == "failed":
        return "end"

    if len(shots_needing_regeneration) > 0:
        logger.info(f"   → Routing to Agent 15A for {len(shots_needing_regeneration)} regenerations")
        return "agent_15A"
    elif len(shots_needing_edit) > 0:
        logger.info(f"   → Shots need editing. Routing to Agent 7 (Shot Editor)")
        return "agent_7"

    # All shots approved by Agent 15 — check if product shots need fidelity review
    product_shot_ids = _get_product_shot_ids(state)
    product_image_url = state.get("product_image_url")

    if product_shot_ids and product_image_url:
        logger.info(f"   → Product shots detected ({len(product_shot_ids)}). Routing to Agent 16 (Product Reviewer)")
        return "product_reviewer"

    _log_product_review_skip(product_shot_ids, product_image_url)
    return "final_checkpoint"


def route_after_agent_7(state: Phase2State) -> Literal["agent_15_review"]:
    """
    After Agent 7: Always route back to Agent 15 for re-review
    """
    logger.info("   → Routing to Agent 15 for re-review of edited shots")
    return "agent_15_review"


def route_after_agent_15_review(state: Phase2State) -> Literal["agent_15A", "agent_7", "product_reviewer", "final_checkpoint"]:
    """
    After Agent 15 Re-review: Route based on whether regeneration, more edits, or product review is needed.
    """
    current_agent = state.get("current_agent", "")
    shots_needing_regeneration = state.get("shots_needing_regeneration", [])
    shots_needing_edit = state.get("shots_needing_edit", [])

    if len(shots_needing_regeneration) > 0:
        logger.info(f"   → Routing to Agent 15A for {len(shots_needing_regeneration)} regenerations")
        return "agent_15A"
    elif current_agent == "agent_7" or len(shots_needing_edit) > 0:
        logger.info("   → More edits needed. Routing back to Agent 7")
        return "agent_7"

    # Edit loop complete — check if product shots need fidelity review
    product_shot_ids = _get_product_shot_ids(state)
    product_image_url = state.get("product_image_url")

    if product_shot_ids and product_image_url:
        logger.info(f"   → Edit loop complete. Product shots detected. Routing to Agent 16 (Product Reviewer)")
        return "product_reviewer"

    _log_product_review_skip(product_shot_ids, product_image_url, prefix="Edit loop complete. ")
    return "final_checkpoint"


def route_after_product_reviewer(state: Phase2State) -> Literal["product_prompt_gen", "final_checkpoint"]:
    """
    After Agent 16: Route to prompt generator if shots still need fixing, else to final_checkpoint.
    """
    shots_needing_fix = state.get("shots_needing_product_fix", [])
    if shots_needing_fix:
        logger.info(f"   → {len(shots_needing_fix)} shots need product fix. Routing to Agent 17")
        return "product_prompt_gen"
    logger.info("   → All product shots approved. Routing to Final Approval Checkpoint")
    return "final_checkpoint"


def route_after_final_approval(state: Phase2State) -> Literal["end", "wait"]:
    """
    After Final Approval: Route based on human decision
    """
    pipeline_status = state.get("pipeline_status", "")
    
    if pipeline_status == "waiting_for_final_approval":
        return "wait"
    elif pipeline_status in ["completed", "rejected"]:
        return "end"
    
    return "wait"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def create_phase2_workflow(starting_node: str = "agent_1"):
    """
    Creates the LangGraph workflow for phase 2 agents with human approval
    Extended to include Agent 12 (Shot Design), Agent 13 (Prompt Modifier),
    Agent 14 (Imagen Generator), Agent 15 (Image Reviewer), Agent 7 (Shot Editor),
    and Edit-Review Loop with Final Human Approval
    """
    # Create state graph
    workflow = StateGraph(Phase2State)

    # Add all nodes
    workflow.add_node("agent_1", agent_1_strategy_node)
    workflow.add_node("human_checkpoint", human_approval_checkpoint)
    workflow.add_node("agent_2", agent_2_prompt_generator_node)
    workflow.add_node("agent_3", agent_3_prompt_review_node)
    workflow.add_node("agent_12", agent_12_shot_design_node)
    workflow.add_node("agent_13", agent_13_prompt_modifier_node)
    workflow.add_node("prompt_checkpoint", prompt_approval_checkpoint)
    workflow.add_node("agent_14", agent_14_imagen_generator_node)
    workflow.add_node("agent_15", agent_15_image_reviewer_node)
    workflow.add_node("agent_15A", agent_15A_prompt_regeneration_node)
    workflow.add_node("agent_14_regenerate", agent_14_regenerate_node)
    workflow.add_node("agent_7", agent_7_shot_editor_node)
    workflow.add_node("agent_15_review", agent_15_review_loop_node)
    workflow.add_node("product_reviewer", agent_16_product_reviewer_node)
    workflow.add_node("product_prompt_gen", agent_17_product_prompt_gen_node)
    workflow.add_node("product_editor", agent_18_product_editor_node)
    workflow.add_node("final_checkpoint", final_approval_checkpoint)

    valid_entry_nodes = {
        "agent_1",
        "human_checkpoint",
        "agent_2",
        "agent_3",
        "agent_12",
        "agent_13",
        "prompt_checkpoint",
        "agent_14",
        "agent_15",
        "agent_15A",
        "agent_14_regenerate",
        "agent_7",
        "agent_15_review",
        "product_reviewer",
        "product_prompt_gen",
        "product_editor",
        "final_checkpoint",
    }
    entry_node = starting_node if starting_node in valid_entry_nodes else "agent_1"

    # Set entry point - will be determined by caller
    workflow.set_entry_point(entry_node)

    # Add edges
    # Agent 1 → Human Checkpoint (always)
    workflow.add_edge("agent_1", "human_checkpoint")

    # Human Checkpoint → Agent 2 (if approved) OR Wait (if pending) OR End (if rejected)
    workflow.add_conditional_edges(
        "human_checkpoint",
        route_after_approval,
        {
            "agent_2": "agent_2",
            "wait": END,  # Pause workflow
            "end": END,
        }
    )

    # Agent 2 → Agent 3 (always)
    workflow.add_edge("agent_2", "agent_3")

    # Agent 3 → Agent 12 (always)
    workflow.add_edge("agent_3", "agent_12")

    # Agent 12 → Agent 13 (always)
    workflow.add_edge("agent_12", "agent_13")

    # Agent 13 → Prompt Checkpoint (always)
    workflow.add_edge("agent_13", "prompt_checkpoint")

    # Prompt Checkpoint → Agent 14 (if approved) OR Wait (if pending) OR End (if rejected)
    workflow.add_conditional_edges(
        "prompt_checkpoint",
        route_after_prompt_approval,
        {
            "agent_14": "agent_14",
            "wait": END,  # Pause workflow
            "end": END,
        }
    )

    # Agent 14 → Agent 15 (always)
    workflow.add_edge("agent_14", "agent_15")

    # Agent 15 → Agent 15A / Agent 7 / Product Reviewer / Final Checkpoint
    workflow.add_conditional_edges(
        "agent_15",
        route_after_agent_15,
        {
            "agent_15A": "agent_15A",
            "agent_7": "agent_7",
            "product_reviewer": "product_reviewer",
            "final_checkpoint": "final_checkpoint",
            "end": END,
        }
    )

    # Agent 15A → Agent 14 Regenerate (always - for regeneration path)
    workflow.add_edge("agent_15A", "agent_14_regenerate")

    # Agent 14 Regenerate → Agent 15 (always - back to review)
    workflow.add_edge("agent_14_regenerate", "agent_15")

    # Agent 7 → Agent 15 Review (always - for re-review)
    workflow.add_conditional_edges(
        "agent_7",
        route_after_agent_7,
        {
            "agent_15_review": "agent_15_review",
        }
    )

    # Agent 15 Review → Agent 15A / Agent 7 / Product Reviewer / Final Checkpoint
    workflow.add_conditional_edges(
        "agent_15_review",
        route_after_agent_15_review,
        {
            "agent_15A": "agent_15A",
            "agent_7": "agent_7",
            "product_reviewer": "product_reviewer",
            "final_checkpoint": "final_checkpoint",
        }
    )

    # Product Reviewer → Product Prompt Gen (if shots fail) OR Final Checkpoint (if all pass)
    workflow.add_conditional_edges(
        "product_reviewer",
        route_after_product_reviewer,
        {
            "product_prompt_gen": "product_prompt_gen",
            "final_checkpoint": "final_checkpoint",
        }
    )

    # Product Prompt Gen → Product Editor (always)
    workflow.add_edge("product_prompt_gen", "product_editor")

    # Product Editor → Product Reviewer (always — back for re-review)
    workflow.add_edge("product_editor", "product_reviewer")

    # Final Checkpoint → Wait (if pending approval) OR End (if approved/rejected)
    workflow.add_conditional_edges(
        "final_checkpoint",
        route_after_final_approval,
        {
            "wait": END,  # Pause workflow
            "end": END,
        }
    )

    # Compile the graph
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_phase2_pipeline(
    shot_list_request: Dict[str, Any],
    show_id: str,
    episode_number: int,
    scene_description: str = None,
    mongodb_client = None,
    strategy_approval: bool = None,
    strategy_feedback: Dict[str, Any] = None,
    project_id: str = None,
    job_id: str = None,
    movie_id: str = None,
    v1_project_id: str = None,
) -> Dict[str, Any]:
    """
    Run the complete Phase 2 pipeline using LangGraph

    Args:
        shot_list_request: Shot list request data
        show_id: Show identifier
        episode_number: Episode number
        scene_description: Optional scene description
        mongodb_client: MongoDB client instance
        strategy_approval: Human approval decision (True/False)
        strategy_feedback: Human feedback for strategies
        project_id: Phase 1 project_id for loading assets (optional)
        job_id: Pipeline job identifier for tracking (optional)
        movie_id: Movie ID to fetch visual_style from movies collection (optional)

    Returns:
        Final state dictionary with all results
    """
    logger.info("\n" + "🎉 "*40)
    logger.info("PHASE 2 LANGGRAPH PIPELINE STARTING")
    logger.info("🎉 "*40 + "\n")

    # Determine starting agent based on strategy approval
    if strategy_approval is True:
        logger.info("Strategies already approved - starting from Agent 2 (Image Prompt Generator)")
        starting_agent = "agent_2"
        agent1_status = "completed"  # Mark as completed since it was already run
    else:
        logger.info("Starting from Agent 1 (Shot Strategy Agent)")
        starting_agent = "agent_1"
        agent1_status = "pending"

    if not movie_id:
        raise ValueError("movie_id is required to fetch visual_style for Phase 2 pipeline")

    # Fetch product image URL from production MongoDB.
    # In master movie workflows, show_id is the scene project id where the product image is saved.
    # project_id remains a fallback for direct Phase 2 callers.
    product_image_url = None
    shots = shot_list_request.get("shots", []) if isinstance(shot_list_request, dict) else []
    has_product_shots = any(s.get("product_present") for s in shots)
    if has_product_shots:
        product_image_url = _fetch_project_product_image_url(show_id)
        if product_image_url:
            logger.info(f"✓ Phase 2: product image URL fetched from scene project {show_id}")
        elif project_id and project_id != show_id:
            product_image_url = _fetch_project_product_image_url(project_id)
            if product_image_url:
                logger.info(f"✓ Phase 2: product image URL fetched from fallback project {project_id}")
            else:
                logger.warning(
                    f"Phase 2: product_present shots found but neither scene project {show_id} "
                    f"nor fallback project {project_id} has product_image_s3_url"
                )
        else:
            logger.warning(
                f"Phase 2: product_present shots found but scene project {show_id} has no product_image_s3_url"
            )

    # Initialize state
    initial_state: Phase2State = {
        "shot_list_request": shot_list_request,
        "show_id": show_id,
        "episode_number": episode_number,
        "project_id": project_id,
        "scene_description": scene_description,
        "job_id": job_id,
        "movie_id": movie_id,
        "annotated_shot_list": None,
        "strategy_analysis_results": {},
        "agent1_status": agent1_status,
        "agent1_human_feedback": None,
        "strategy_approval_decision": strategy_approval,
        "strategy_approval_feedback": strategy_feedback,
        "image_prompts_generated": {},
        "agent2_status": "pending",
        "agent2_human_feedback": None,
        "reviewed_prompts": {},
        "agent3_status": "pending",
        "agent3_human_feedback": None,
        "shot_designs": {},
        "agent12_status": "pending",
        "agent12_human_feedback": None,
        "modified_prompts": {},
        "agent13_status": "pending",
        "agent13_human_feedback": None,
        "prompt_approval_decision": None,
        "prompt_approval_feedback": None,
        "generated_images": {},
        "agent14_status": "pending",
        "agent14_human_feedback": None,
        "image_reviews": {},
        "agent15_status": "pending",
        "agent15_human_feedback": None,
        "regenerated_prompts": {},
        "agent15A_status": "pending",
        "agent15A_human_feedback": None,
        "edited_shots": {},
        "agent7_status": "pending",
        "agent7_human_feedback": None,
        "edit_loop_iterations": {},
        "shots_needing_edit": [],
        "shots_approved": [],
        "shots_max_retries": [],
        "regenerate_loop_iterations": {},
        "shots_needing_regeneration": [],
        "shots_edit_instructions": {},
        "product_review_results": {},
        "product_review_iterations": {},
        "product_fix_prompts": {},
        "product_corrected_images": {},
        "shots_needing_product_fix": [],
        "shots_product_approved": [],
        "final_approval_decision": None,
        "final_approval_feedback": None,
        "mongodb_client": mongodb_client,
        "mongodb_operations": {},
        "current_agent": starting_agent,
        "pipeline_status": "running",
        "error_message": None,
        "output_files": [],
        "requires_human_feedback": False,
        "feedback_agent": None,
        "episode_id": shot_list_request.get("episode_id", ""),
        "title": shot_list_request.get("title", ""),
        "v1_project_id": v1_project_id,
        "product_image_url": product_image_url,
    }

    # If starting from agent_2, we need to load the existing annotated shots from MongoDB
    if strategy_approval is True and mongodb_client:
        try:
            logger.info("📥 Loading existing annotated shots from MongoDB...")
            shot_collection = mongodb_client.get_shots_from_atlas(show_id, episode_number)
            if shot_collection:
                # Convert MongoDB data back to AnnotatedShotList
                from app.models.mongodb.shots import AnnotatedShotList, AnnotatedShotItem
                
                annotated_shots = []
                for shot_data in shot_collection.annotated_shots:
                    shot_item = AnnotatedShotItem(**shot_data.model_dump())
                    annotated_shots.append(shot_item)
                
                annotated_shot_list = AnnotatedShotList(
                    episode_id=shot_collection.episode_id,
                    title=shot_collection.title,
                    annotated_shots=annotated_shots,
                    overall_continuity_notes=shot_collection.overall_continuity_notes,
                    strategy_summary=shot_collection.strategy_summary
                )
                
                initial_state["annotated_shot_list"] = annotated_shot_list
                initial_state["strategy_analysis_results"] = annotated_shot_list.model_dump() if hasattr(annotated_shot_list, 'model_dump') else annotated_shot_list
                logger.info(f"Loaded {len(annotated_shots)} annotated shots from MongoDB")
            else:
                raise ValueError(f"No shots found in MongoDB for show_id: {show_id}, episode_number: {episode_number}")
        except Exception as e:
            logger.error(f"Failed to load shots from MongoDB: {e}")
            raise ValueError(f"Cannot start from Agent 2 without existing data: {e}")

    # Create and run workflow
    app = create_phase2_workflow(starting_agent)

    # Execute the workflow
    final_state = app.invoke(initial_state)

    # Print results
    logger.info("\n" + "🎉 "*40)
    logger.info("PHASE 2 PIPELINE COMPLETED")
    logger.info("🎉 "*40)

    logger.info(f"\nPIPELINE STATUS: {final_state['pipeline_status']}")
    logger.info(f"\n📁 OUTPUT FILES:")
    for file in final_state.get("output_files", []):
        logger.info(f"   • {file}")

    if final_state["pipeline_status"] == "failed":
        logger.error(f"\nERROR: {final_state.get('error_message', 'Unknown error')}")

    return final_state


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv

    load_dotenv()

    # Get local mongouri from environment or local defaults
    import os
    from backend.services.production.app.models.mongodb.shots import MongoClient

    # Try setting up a mock MongoDB client to satisfy save validations. 
    # For now we'll pass none and let it gracefully skip.
    client = None

    # Example shot list request
    example_shot_list = {
        "episode_id": "E01",
        "title": "Test Episode",
        "shots": [
            {
                "shot_id": "S001",
                "description": "Wide shot of the character walking down the street",
                "duration": 3.0,
                "scene_number": 1,
                "sequence_number": 1,
                "shot_style": "wide_shot",
                "camera_movement": "static"
            }
        ]
    }

    # Run pipeline end-to-end bypassing human checkpoints
    result = run_phase2_pipeline(
        shot_list_request=example_shot_list,
        show_id="test_show",
        episode_number=1,
        scene_description="A character walking scene",
        movie_id="test_movie",
        strategy_approval=True
    )

    logger.info("\nPipeline execution complete!")
