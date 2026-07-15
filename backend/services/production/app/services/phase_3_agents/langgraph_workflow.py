#!/usr/bin/env python3
"""
LangGraph Workflow Orchestrator for Phase 3 Agents
===================================================
Coordinates the video generation pipeline using LangGraph state management.

Flow:
MongoDB Shot → Prompt Selection (A or B) → Video Generation → AI Review →
Human Checkpoint → Final Output

Conditional Logic:
1. If generation_strategy is "generate_new" or "last_frame_seed" → use video_prompt_A
2. If generation_strategy is "multi_shot" → use video_prompt_B
3. After video generation → AI review
4. If AI review not approved → regenerate (max 3 times)
5. If AI review approved → human checkpoint
6. If human needs_changes → regenerate with updated prompt (max 3 times)
7. If human approved → END
"""


import asyncio
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional, Tuple

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

logger = get_logger(__name__)
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
services_dir = os.path.join(current_dir, '..')
sys.path.insert(0, services_dir)

from phase_3_agents.workflow_state import Phase3State
from backend.services.production.app.config import get_mongo_factory
from backend.services.production.app.services.project_service import ProjectService
from backend.services.production.app.services.pipeline_service import PipelineService
from backend.shared.utils.mongodb_validators import validate_object_id

# Import Phase 3 agents
from phase_3_agents.video_generation.video_generation_api_agent import VideoGenerationAPIAgent
from phase_3_agents.video_generation.video_model import resolve_video_model
from phase_3_agents.video_generation.video_review_agent import VideoReviewAgent
from phase_3_agents.video_prompt_A.agent_video_generation import VideoGenerationAgent
from phase_3_agents.video_prompt_A.video_prompt_review_A_agent.video_prompt_review_A_agent import VideoPromptReviewAgent
from phase_3_agents.video_prompt_B.video_prompt_B import MultiShotVideoGenerator


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_scene_sequence_from_shot_id(shot_id: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse scene_number and sequence_number from shot_id.
    Format: S01E{scene:02d}_{seq:03d}  e.g. S01E01_001 → (1, 1)
    """
    match = re.match(r'^S\d+E(\d+)_(\d+)', shot_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None



def fetch_shot_from_mongodb(shot_id: str, show_id: str = None) -> Dict[str, Any]:
    """
    Fetch shot data from MongoDB by shot_id and show_id from shots collection with production_projects fallback

    Priority:
    1. shots collection - annotated_shots array (episode-based structure)
    2. shots collection - individual shot document
    3. production_projects collection - agent14/agent12 outputs (fallback for testing)

    Args:
        shot_id: Shot ID to fetch
        show_id: Show/Project ID to uniquely identify the shot (required to avoid conflicts)

    Returns:
        Shot document from MongoDB
    """
    mongo_factory = get_mongo_factory()
    client, shots_collection = mongo_factory.get_collection("shots")

    try:
        # Priority 1: Look for shot in annotated_shots array, filtered by show_id
        query = {"annotated_shots.shot_id": shot_id}
        if show_id:
            query["show_id"] = show_id

        episode_doc = shots_collection.find_one(query)

        if episode_doc and "annotated_shots" in episode_doc:
            # Find the specific shot in the annotated_shots array
            for shot in episode_doc["annotated_shots"]:
                if shot.get("shot_id") == shot_id:
                    # Add episode context to the shot
                    shot_with_context = {
                        **shot,
                        "episode_id": episode_doc.get("_id"),
                        "episode_title": episode_doc.get("title"),
                        "episode_description": episode_doc.get("scene_description"),
                        "show_id": episode_doc.get("show_id"),
                        "episode_number": episode_doc.get("episode_number")
                    }
                    logger.info(f"Found shot {shot_id} in shots collection (annotated_shots array)")
                    return shot_with_context

        # Priority 2: Fallback to individual shot document (current structure)
        if show_id:
            # Try different possible show_id field names
            query = {
                "shot_id": shot_id,
                "$or": [
                    {"show_id": show_id},
                    {"project_id": show_id}
                ]
            }
        else:
            query = {"shot_id": shot_id}

        shot = shots_collection.find_one(query)
        if shot:
            logger.info(f"Found shot {shot_id} in shots collection (individual document)")
            return shot

        # Priority 3: Fallback to production_projects collection
        if show_id:
            logger.warning(f"Shot {shot_id} not found in shots collection, trying production_projects fallback for show {show_id}...")
        else:
            logger.warning(f"Shot {shot_id} not found in shots collection, trying production_projects fallback (no show_id provided - may return wrong shot if duplicates exist)...")

        client, projects_collection = mongo_factory.get_collection("production_projects")

        # Build query - require show_id for safety
        if show_id:
            try:
                show_id_obj = validate_object_id(show_id)
                query = {"_id": show_id_obj}
                project_doc = projects_collection.find_one(query)
            except ValueError as e:
                logger.error(f"Invalid show_id format: {e}")
                logger.error(f"Invalid show_id format: {e}")
                # Try as string fallback for legacy compatibility
                query = {"_id": show_id}
                project_doc = projects_collection.find_one(query)
        else:
            # WARNING: Without show_id, we might get the wrong shot if duplicates exist
            project_doc = projects_collection.find_one({
                "agent_outputs.agent14.output.generated_images.shot_id": shot_id
            })

        if project_doc:
            # Extract shot from agent14 output
            agent14_output = project_doc.get("agent_outputs", {}).get("agent14", {}).get("output", {})
            generated_images = agent14_output.get("generated_images", [])

            # Also get agent12 output for shot designs
            agent12_output = project_doc.get("agent_outputs", {}).get("agent12", {}).get("output", {})
            shot_designs = {sd["shot_id"]: sd for sd in agent12_output.get("shot_designs", [])}

            # Find the shot
            for img in generated_images:
                if img.get("shot_id") == shot_id:
                    shot_design = shot_designs.get(shot_id, {})

                    # Build shot document similar to shots collection structure
                    shot_doc = {
                        "shot_id": shot_id,
                        "image_url": img.get("s3_url"),
                        "local_path": img.get("local_path"),
                        "prompt": img.get("prompt"),
                        "assets_used": img.get("assets_used", []),
                        "generation_timestamp": img.get("generation_timestamp"),
                        "metadata": img.get("metadata", {}),

                        # Add shot design info
                        "generation_strategy": shot_design.get("generation_strategy", "generate_new"),
                        "selected_assets": shot_design.get("selected_assets", []),
                        "model_recommendation": shot_design.get("model_recommendation"),
                        "composition_strategy": shot_design.get("composition_strategy", {}),
                        "shot_metadata": shot_design.get("metadata", {}),

                        # Project context
                        "project_id": str(project_doc.get("_id")),
                        "project_name": project_doc.get("name"),
                        "show_id": str(project_doc.get("_id")),
                        "script": project_doc.get("script"),

                        # Mark source
                        "_source": "production_projects_fallback"
                    }

                    if show_id:
                        logger.info(f"Found shot {shot_id} in production_projects collection (fallback) for show {show_id}")
                    else:
                        logger.warning(f"Found shot {shot_id} in production_projects collection (fallback) - NO show_id validation!")
                    return shot_doc

        if show_id:
            raise ValueError(f"Shot {shot_id} not found in shots or production_projects collections for show {show_id}")
        else:
            raise ValueError(f"Shot {shot_id} not found in shots or production_projects collections")
    finally:
        pass  # Don't close singleton client


def save_video_to_mongodb(shot_id: str, version: str, video_data: Dict[str, Any], show_id: str = None) -> bool:
    """
    Save video generation results to MongoDB in the video field of annotated_shots array.

    Uses a two-step approach with arrayFilters to handle null video fields:
    1. Initialize video field to {} if it's null
    2. Set the video version data

    This is necessary because MongoDB cannot create nested fields (like video.v0)
    when the parent field (video) is explicitly null.

    Args:
        shot_id: Shot ID
        version: Version string (v0, v1, v2)
        video_data: Video data to save
        show_id: Show/Project ID to uniquely identify the shot (CRITICAL to avoid updating wrong shot)

    Returns:
        bool: True if save successful, False otherwise
    """
    from datetime import datetime

    mongo_factory = get_mongo_factory()
    client, shots_collection = mongo_factory.get_collection("shots")

    try:
        # Ensure video_data has all required fields
        if "approval_status" not in video_data:
            video_data["approval_status"] = "pending"
        if "approval_feedback" not in video_data:
            video_data["approval_feedback"] = ""
        if "approved_at" not in video_data:
            video_data["approved_at"] = None

        logger.info(f"Attempting to save video {version} for shot {shot_id} to shots collection...")

        # Build query filter - MUST include show_id to avoid updating wrong shot
        # (shot_ids are NOT unique across different shows!)
        query_filter = {"annotated_shots.shot_id": shot_id}
        if show_id:
            query_filter["show_id"] = show_id
            logger.info(f"   Using show_id filter: {show_id}")
        else:
            logger.warning(f"   ⚠️  No show_id provided! This may update the wrong shot if there are duplicates.")

        # Step 1: Initialize video field to {} if it's null
        # This is required because MongoDB cannot create nested fields when parent is null
        init_result = shots_collection.update_one(
            query_filter,
            {"$set": {"annotated_shots.$[elem].video": {}}},
            array_filters=[{"elem.shot_id": shot_id, "elem.video": None}]
        )

        if init_result.modified_count > 0:
            logger.info(f"   Initialized video field from null to {{}}")

        # Step 2: Set the video version data using arrayFilters
        result = shots_collection.update_one(
            query_filter,
            {
                "$set": {
                    f"annotated_shots.$[elem].video.{version}": video_data,
                    "annotated_shots.$[elem].updated_at": datetime.now()
                }
            },
            array_filters=[{"elem.shot_id": shot_id}]
        )

        if result.matched_count > 0:
            if result.modified_count > 0:
                logger.info(f"✅ Successfully saved video {version} to shots collection for shot {shot_id}")
                return True
            else:
                logger.warning(f"⚠️  Shot {shot_id} found but not modified (data may be identical)")
                return True  # Still considered success if matched
        else:
            logger.error(f"❌ No document found with shot_id {shot_id} in annotated_shots array")
            return False

    except Exception as e:
        logger.error(f"❌ Exception while saving video to shots collection: {e}", exc_info=True)
        return False
    finally:
        pass  # Don't close singleton client


# ============================================================================
# NODE FUNCTIONS - Each agent as a LangGraph node
# ============================================================================

def initialize_node(state: Phase3State) -> Phase3State:
    """
    Initialize Node: Fetch shot data from MongoDB and set up initial state

    If human_decision is present this is a resume from a human checkpoint.
    shot_data is re-fetched when empty (run_phase3_pipeline always initialises
    it as {} so it cannot be used as the resume sentinel).
    """
    # Detect resume: human_decision is only set when the caller is continuing
    # after a human review pause — shot_data is irrelevant as the sentinel.
    if state.get("human_decision"):
        logger.info("\n" + "="*60)
        logger.info("RESUMING FROM HUMAN CHECKPOINT")
        logger.info("="*60)
        logger.info(f"Shot ID: {state['shot_id']}")
        logger.info(f"Human Decision: {state.get('human_decision')}")
        logger.info(f"Current Version: v{state.get('current_version', 0)}")

        # Re-fetch shot_data if it was not carried over (fresh call via run_phase3_pipeline)
        shot_data = state.get("shot_data") or {}
        if not shot_data:
            try:
                show_id = state.get("show_id")
                shot_data = fetch_shot_from_mongodb(state["shot_id"], show_id=show_id)
                logger.info("Re-fetched shot data for resumed checkpoint")
            except Exception as e:
                logger.error(f"Failed to re-fetch shot data on resume: {e}")
                return {
                    **state,
                    "pipeline_status": "failed",
                    "error_message": str(e),
                }

        return {
            **state,
            "shot_data": shot_data,
            "pipeline_status": "running",
            "current_node": "human_checkpoint",
        }

    logger.info("\n" + "="*60)
    logger.info("INITIALIZING PHASE 3 WORKFLOW")
    logger.info("="*60)
    logger.info(f"Shot ID: {state['shot_id']}")

    try:
        # Fetch shot data from MongoDB (with optional show_id for fallback)
        show_id = state.get("show_id")
        shot_data = fetch_shot_from_mongodb(state["shot_id"], show_id=show_id)

        generation_strategy = shot_data.get("generation_strategy", "generate_new")

        # Resolve scene/sequence from DB fields; fall back to parsing the shot_id
        _parsed_scene, _parsed_seq = parse_scene_sequence_from_shot_id(state["shot_id"])
        scene_number = shot_data.get("scene_number") or _parsed_scene
        sequence_number = shot_data.get("sequence_number") or _parsed_seq

        logger.info(f"Loaded shot data from MongoDB")
        logger.info(f"   Strategy: {generation_strategy}")
        logger.info(f"   Description: {shot_data.get('description', 'N/A')[:100]}...")
        logger.info(f"   Scene: {scene_number}, Shot: {sequence_number}")

        return {
            **state,
            "shot_data": shot_data,
            "generation_strategy": generation_strategy,
            "show_id": shot_data.get("show_id", ""),
            "episode_number": shot_data.get("episode_number", 1),
            "scene_number": scene_number,
            "sequence_number": sequence_number,
            "video_generation_attempt": 0,
            "max_video_generation_attempts": 3,
            "human_regeneration_attempt": 0,
            "max_human_regeneration_attempts": 3,
            "current_version": 0,
            "video_versions": [],
            "current_node": "prompt_router",
            "pipeline_status": "running",
        }

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return {
            **state,
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def prompt_router_node(state: Phase3State) -> Phase3State:
    """
    Prompt Router Node: Generate video prompt based on generation strategy

    - generate_new or last_frame_seed → use VideoGenerationAgent (Prompt A)
    - multi_shot → use MultiShotVideoGenerator (Prompt B)
    """
    logger.info("\n" + "="*60)
    logger.info("🔀 VIDEO PROMPT GENERATION")
    logger.info("="*60)

    try:
        shot_data = state["shot_data"]
        generation_strategy = state["generation_strategy"]
        shot_id = state["shot_id"]

        logger.info(f"Strategy: {generation_strategy}")
        logger.info(f"Generating video prompt for shot: {shot_id}")

        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY")

        # Get ShotsService for fetching seed shots (using singleton connection)
        from app.config import get_shots_service
        shots_service = get_shots_service()

        # Generate prompt based on strategy
        if generation_strategy in ["generate_new", "last_frame_seed"]:
            # Use VideoGenerationAgent for Prompt A
            logger.info(f"Using VideoGenerationAgent (Prompt A)")
            agent = VideoGenerationAgent(api_key=api_key, enable_saving=False)

            # Generate prompt
            result = agent.generate_video_prompt(
                shot=shot_data,
                scene_description=shot_data.get("description", ""),
                mongodb_client=shots_service
            )

            video_prompt = result.get("video_prompt", "")
            prompt_version = "A"

        elif generation_strategy == "multi_shot":
            # Use MultiShotVideoGenerator for Prompt B
            logger.info(f"Using MultiShotVideoGenerator (Prompt B)")
            agent = MultiShotVideoGenerator(api_key=api_key)

            # Generate prompt - just need shot description
            shot_description = shot_data.get("description", "")
            reference_context = shot_data.get("reference_context", "")

            video_prompt = agent.generate_video_prompt(
                shot_description=shot_description,
                reference_context=reference_context if reference_context else None
            )
            prompt_version = "B"
        else:
            raise ValueError(f"Unknown generation strategy: {generation_strategy}")

        logger.info(f"Generated Prompt {prompt_version}")
        if video_prompt:
            logger.info(f"   Prompt length: {len(video_prompt)} chars")
            logger.info(f"   Prompt preview: {video_prompt[:100]}...")
        else:
            logger.warning(f"   WARNING: Prompt is empty!")
            raise ValueError("Video prompt generation returned empty prompt")

        # Run prompt through VideoPromptReviewAgent (audio safety + Veo structure check)
        logger.info(f"Running video prompt review...")
        try:
            review_agent = VideoPromptReviewAgent(api_key=api_key, enable_saving=False)
            shot_with_draft = {**shot_data, "prompt_video_draft": video_prompt}
            review_result = asyncio.run(
                review_agent.review_prompt_with_gemini(
                    shot=shot_with_draft,
                    scene_description=shot_data.get("description", "")
                )
            )
            reviewed_prompt = review_result.get("updated_prompt", "").strip()
            if reviewed_prompt and reviewed_prompt != video_prompt:
                logger.info(f"   ✏️ Reviewer updated prompt")
                logger.info(f"   Changes: {review_result.get('changes_made', 'N/A')}")
                logger.info(f"   Reviewed prompt length: {len(reviewed_prompt)} chars")
                video_prompt = reviewed_prompt
            else:
                logger.info(f"   ✅ Prompt passed review unchanged")
        except Exception as e:
            logger.warning(f"   ⚠️ Prompt review failed — using original prompt: {e}")

        # Save agent17 output to production_projects
        if state.get("show_id"):
            try:
                project_service = ProjectService()
                project_service.update_agent_output(
                    project_id=state["show_id"],
                    agent_number=17,
                    status="completed",
                    output={
                        "video_prompt": video_prompt,
                        "prompt_version": prompt_version,
                        "generation_strategy": generation_strategy,
                        "shot_id": shot_id
                    }
                )
                logger.info(f"   💾 Saved agent17 output to production_projects")
            except Exception as e:
                logger.warning(f"   Failed to save agent17 output: {e}")

        # Update pipeline job status - agent_17 completed, moving to agent_18
        if state.get("job_id"):
            try:
                pipeline_service = PipelineService()
                pipeline_service.update_job_state(
                    state["job_id"],
                    {
                        "current_agent": "agent_18",
                        "agent17_status": "completed",
                        "agent18_status": "running"
                    }
                )
                logger.info(f"   💾 Updated job status: agent_17 completed, agent_18 starting")
            except Exception as e:
                logger.warning(f"   Failed to update job status: {e}")

        return {
            **state,
            "video_prompt": video_prompt,
            "prompt_version": prompt_version,
            "current_node": "video_generation",
        }

    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def _patch_prompt_for_audio_safety(prompt: str, attempt: int) -> str:
    """
    Harden a prompt that was blocked by Veo's RAI audio safety filter.
    Attempt 1: append silence note (covers most cases).
    Attempt 2+: also replace known auditory action verbs.
    """
    import re
    SILENCE_NOTE = "SFX: complete silence, gentle room tone only."
    if SILENCE_NOTE not in prompt:
        prompt = f"{prompt} {SILENCE_NOTE}"
    if attempt >= 2:
        replacements = {
            r"\btaps\b": "touches",
            r"\bsips\b": "holds",
            r"\bsip\b": "hold",
            r"\bdrinks\b": "holds",
            r"\bsighs\b": "exhales",
            r"\bwhispers\b": "says",
            r"\bbreathes\b": "inhales",
            r"\bgasps\b": "reacts",
            r"\bmoans\b": "says",
        }
        for pattern, replacement in replacements.items():
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
    return prompt


def video_generation_node(state: Phase3State) -> Phase3State:
    """
    Video Generation Node: Generate video using Gemini Veo 3.1
    """
    logger.info("\n" + "="*60)
    logger.info("🎥 VIDEO GENERATION")
    logger.info("="*60)
    logger.info(f"Attempt: {state['video_generation_attempt'] + 1}/{state['max_video_generation_attempts']}")

    # Update pipeline job status - agent_18 starting (in case of regeneration)
    if state.get("job_id") and state.get("video_generation_attempt", 0) > 0:
        try:
            pipeline_service = PipelineService()
            pipeline_service.update_job_state(
                state["job_id"],
                {
                    "current_agent": "agent_18",
                    "agent18_status": "running"
                }
            )
            logger.info(f"   💾 Updated job status: agent_18 regenerating (attempt {state['video_generation_attempt'] + 1})")
        except Exception as e:
            logger.warning(f"   Failed to update job status: {e}")

    try:
        # Initialize video generation agent, using the model configured on the
        # movie (movies.global_settings.video_model), resolved via show_id.
        api_key = os.getenv("GOOGLE_API_KEY")
        video_model = resolve_video_model(state.get("show_id", ""))
        agent = VideoGenerationAPIAgent(api_key=api_key, video_model=video_model)

        shot_data = state["shot_data"]

        # Get start image URL
        mongo_factory = get_mongo_factory()
        client, shots_collection = mongo_factory.get_collection("shots")

        try:
            # Get image version from state if specified
            image_version = state.get("image_version")
            start_image_url = agent.fetch_start_image_url(shot_data, shots_collection, image_version)

            if not start_image_url:
                raise ValueError("Could not fetch start image URL")

            logger.info(f"Start image: {start_image_url}")

            # Get video prompt (use human updated prompt if available)
            if state.get("human_updated_prompt"):
                video_prompt = state["human_updated_prompt"]
                logger.info(f"Using human-updated prompt")
            elif state.get("suggested_prompt") and state["video_generation_attempt"] > 0:
                video_prompt = state["suggested_prompt"]
                logger.info(f"Using AI-suggested prompt")
            else:
                video_prompt = state["video_prompt"]
                logger.info(f"Using original prompt")

            # Generate video (with RAI audio filter retry)
            logger.info(f"Generating video...")
            MAX_RAI_RETRIES = 2
            rai_attempt = 0
            active_prompt = video_prompt
            result = None

            while True:
                result = agent.generate_video(
                    shot=shot_data,
                    video_prompt=active_prompt,
                    start_image_url=start_image_url,
                    mongodb_client=shots_collection,
                    scene_number=state.get("scene_number"),
                    sequence_number=state.get("sequence_number"),
                    version=state["current_version"] + 1,
                    show_id=state.get("show_id", ""),
                )

                if result and result.get("status") == "rai_filtered":
                    if rai_attempt < MAX_RAI_RETRIES:
                        rai_attempt += 1
                        logger.warning(
                            f"⚠️ RAI audio filter hit — patching prompt and retrying "
                            f"(RAI attempt {rai_attempt}/{MAX_RAI_RETRIES})"
                        )
                        active_prompt = _patch_prompt_for_audio_safety(active_prompt, rai_attempt)
                        logger.info(f"Patched prompt: {active_prompt}")
                        continue
                    else:
                        logger.error("❌ RAI filter blocked all retry attempts")
                break  # success or unrecoverable error

            if not result or not result.get("video_url"):
                error_msg = result.get("message") if result else "Video generation failed"
                raise ValueError(f"Video generation failed: {error_msg}")

            generated_video_url = result["video_url"]
            task_id = result.get("task_id", "")
            last_frame_url = result.get("last_frame_url")

            logger.info(f"Video generated successfully")
            logger.info(f"   URL: {generated_video_url}")
            if last_frame_url:
                logger.info(f"   Last frame: {last_frame_url}")

            # Track version
            version_str = f"v{state['current_version']}"
            version_data = {
                "updated_prompt": video_prompt,
                "changes_made": state.get("suggested_prompt_reasoning", "Initial generation"),
                "reasoning": f"Video generation attempt {state['video_generation_attempt'] + 1}",
                "generated_videos_s3": [generated_video_url],
                "task_id": task_id,
                "timestamp": result.get("timestamp", ""),
            }

            # Add last frame URL if available
            if last_frame_url:
                version_data["last_frame_s3"] = last_frame_url

            # Save to MongoDB shots collection
            try:
                save_success = save_video_to_mongodb(
                    shot_id=state["shot_id"],
                    version=version_str,
                    video_data=version_data,
                    show_id=state.get("show_id")  # CRITICAL: Pass show_id to avoid updating wrong shot
                )
                if save_success:
                    logger.info(f"   💾 Saved video to shots collection")
                else:
                    logger.error(f"   ❌ Failed to save video to shots collection")
                    # Mark as failed if save fails
                    raise Exception("Failed to save video to shots collection")
            except Exception as e:
                logger.error(f"   ❌ Exception saving video to shots collection: {e}", exc_info=True)
                raise  # Re-raise to trigger proper error handling

            # Update pipeline job status - agent_18 completed, moving to agent_19
            if state.get("job_id"):
                try:
                    pipeline_service = PipelineService()
                    pipeline_service.update_job_state(
                        state["job_id"],
                        {
                            "current_agent": "agent_19",
                            "agent18_status": "completed",
                            "agent19_status": "running"
                        }
                    )
                    logger.info(f"   💾 Updated job status: agent_18 completed, agent_19 starting")
                except Exception as e:
                    logger.warning(f"   Failed to update job status: {e}")

            # Update state
            video_versions = state.get("video_versions", [])
            video_versions.append(version_data)

            return {
                **state,
                "start_image_url": start_image_url,
                "video_generation_task_id": task_id,
                "generated_video_url": generated_video_url,
                "video_generation_status": "completed",
                "video_generation_attempt": state["video_generation_attempt"] + 1,
                "current_node": "ai_review",
                "video_versions": video_versions,
                "current_version": state["current_version"] + 1,
                "mongodb_save_status": "saved",
            }

        finally:
            pass  # Don't close singleton client

    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return {
            **state,
            "video_generation_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def ai_review_node(state: Phase3State) -> Phase3State:
    """
    AI Review Node: Review generated video using VideoReviewAgent
    """
    logger.info("\n" + "="*60)
    logger.info("AI VIDEO REVIEW")
    logger.info("="*60)

    try:
        # Initialize review agent
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = VideoReviewAgent(api_key=api_key)

        # Review the video
        logger.info(f"🎥 Reviewing video: {state['generated_video_url']}")

        shot_data = state["shot_data"]
        review_result = agent.review_video(
            video_url=state["generated_video_url"],
            shot_id=state["shot_id"],
            video_prompt=state["video_prompt"],
            shot_description=shot_data.get("description", ""),
            generation_strategy=state.get("generation_strategy", "")
        )

        if not review_result:
            raise ValueError("Video review failed")

        # review_result is a dict (from model_dump()), not a Pydantic object
        decision = review_result.get("decision")
        score = review_result.get("overall_score")
        prompt_suggestions = review_result.get("prompt_suggestions", {})
        suggested_prompt = prompt_suggestions.get("suggested_prompt", "")
        assessment = review_result.get("assessment", {})
        issues = assessment.get("issues", [])

        logger.info(f"Review completed")
        logger.info(f"   Decision: {decision}")
        logger.info(f"   Score: {score}/100")
        logger.info(f"   Issues: {len(issues)}")

        # Save agent19 output to production_projects
        if state.get("show_id"):
            try:
                project_service = ProjectService()
                project_service.update_agent_output(
                    project_id=state["show_id"],
                    agent_number=19,
                    status="completed",
                    output={
                        "shot_id": state["shot_id"],
                        "video_url": state["generated_video_url"],
                        "decision": decision,
                        "overall_score": score,
                        "review_result": review_result,
                        "suggested_prompt": suggested_prompt,
                        "issues": issues,
                        "timestamp": review_result.get("timestamp", "")
                    },
                    append_output=True  # Append to outputs array for multi-shot processing
                )
                logger.info(f"   💾 Saved agent19 output to production_projects (appended)")
            except Exception as e:
                logger.warning(f"   Failed to save agent19 output: {e}")

        # Update pipeline job status - agent_19 completed
        if state.get("job_id"):
            try:
                pipeline_service = PipelineService()
                pipeline_service.update_job_state(
                    state["job_id"],
                    {
                        "agent19_status": "completed"
                    }
                )
                logger.info(f"   💾 Updated job status: agent_19 completed")
            except Exception as e:
                logger.warning(f"   Failed to update job status: {e}")

        return {
            **state,
            "review_result": review_result,
            "review_decision": decision,
            "review_score": score,
            "suggested_prompt": suggested_prompt,
            "suggested_prompt_reasoning": prompt_suggestions.get("reasoning", ""),
            "ai_review_status": "completed",
            "current_node": "review_decision_router",
        }

    except Exception as e:
        logger.error(f"AI review failed: {e}")
        return {
            **state,
            "ai_review_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def human_checkpoint_node(state: Phase3State) -> Phase3State:
    """
    Human Checkpoint Node: Wait for human approval or changes
    """
    logger.info("\n" + "="*60)
    logger.info("👤 HUMAN CHECKPOINT")
    logger.info("="*60)
    logger.info("⏸Workflow paused. Awaiting human approval...")
    logger.info(f"   Current video: {state['generated_video_url']}")
    logger.info(f"   AI Review score: {state.get('review_score', 0)}/100")
    logger.info(f"   AI Decision: {state.get('review_decision', 'N/A')}")

    # Check if human has already provided a decision
    if state.get("human_decision"):
        decision = state["human_decision"]
        logger.info(f"   ✓ Human decision received: {decision}")

        if decision == "approved":
            return {
                **state,
                "human_checkpoint_status": "completed",
                "pipeline_status": "completed",
                "current_node": "end",
                # Clear human decision fields after processing
                "human_decision": None,
                "human_updated_prompt": None,
                "human_feedback": None,
            }
        elif decision == "needs_changes":
            # Check regeneration limit
            if state["human_regeneration_attempt"] >= state["max_human_regeneration_attempts"]:
                logger.error(f"   Maximum human regeneration attempts reached")
                return {
                    **state,
                    "pipeline_status": "failed",
                    "error_message": f"Maximum human regeneration attempts ({state['max_human_regeneration_attempts']}) exceeded",
                }

            # Go back to video generation with human updated prompt
            return {
                **state,
                "human_checkpoint_status": "completed",
                "current_node": "video_generation",
                "human_regeneration_attempt": state["human_regeneration_attempt"] + 1,
                "video_generation_attempt": 0,  # Reset AI regeneration counter
                # Clear human decision after processing, but keep updated_prompt for video generation
                "human_decision": None,
                "human_feedback": None,
            }

    # No decision yet - pause workflow
    return {
        **state,
        "human_checkpoint_status": "pending",
        "pipeline_status": "waiting_for_human",
        "current_node": "human_checkpoint",
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def route_after_video_generation(state: Phase3State) -> Literal["ai_review", "failed"]:
    """
    Route after video generation based on pipeline status
    """
    if state.get("pipeline_status") == "failed":
        return "failed"
    return "ai_review"

def route_after_ai_review(state: Phase3State) -> Literal["video_generation", "human_checkpoint", "failed"]:
    """
    Route after AI review based on decision and regeneration attempts
    """
    pipeline_status = state.get("pipeline_status", "")

    if pipeline_status == "failed":
        return "failed"

    decision = state.get("review_decision", "")
    attempt = state.get("video_generation_attempt", 0)
    max_attempts = state.get("max_video_generation_attempts", 3)

    if decision == "approved":
        logger.info("   → AI approved! Routing to human checkpoint")
        return "human_checkpoint"
    elif decision in ["refine_prompt", "regenerate"]:
        if attempt >= max_attempts:
            logger.info(f"   → Maximum AI regeneration attempts reached ({max_attempts})")
            logger.info(f"   → Routing to human checkpoint for manual review")
            return "human_checkpoint"
        else:
            logger.info(f"   → AI requested changes. Regenerating ({attempt}/{max_attempts})")
            return "video_generation"
    else:
        # Unknown decision, go to human checkpoint
        logger.info(f"   → Unknown AI decision: {decision}. Routing to human checkpoint")
        return "human_checkpoint"


def route_after_initialize(state: Phase3State) -> Literal["prompt_router", "human_checkpoint"]:
    """
    Route after initialize based on whether this is a new workflow or resuming from checkpoint
    """
    current_node = state.get("current_node", "prompt_router")

    if current_node == "human_checkpoint":
        logger.info("   → Resuming from human checkpoint")
        return "human_checkpoint"
    else:
        logger.info("   → Starting new workflow")
        return "prompt_router"


def route_after_human_checkpoint(state: Phase3State) -> Literal["video_generation", "end", "wait", "failed"]:
    """
    Route after human checkpoint based on decision
    """
    pipeline_status = state.get("pipeline_status", "")

    if pipeline_status == "failed":
        return "failed"

    if pipeline_status == "completed":
        return "end"

    if pipeline_status == "waiting_for_human":
        return "wait"

    current_node = state.get("current_node", "")

    if current_node == "video_generation":
        logger.info("   → Human requested changes. Routing to video generation")
        return "video_generation"
    elif current_node == "end":
        logger.info("   → Human approved! Workflow complete")
        return "end"

    return "wait"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def create_phase3_workflow() -> CompiledStateGraph:
    """
    Creates the LangGraph workflow for Phase 3 video generation pipeline
    """
    # Create state graph
    workflow = StateGraph(Phase3State)

    # Add all nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("prompt_router", prompt_router_node)
    workflow.add_node("video_generation", video_generation_node)
    workflow.add_node("ai_review", ai_review_node)
    workflow.add_node("human_checkpoint", human_checkpoint_node)

    # Set entry point
    workflow.set_entry_point("initialize")

    # Conditional routing from initialize (for resume support)
    workflow.add_conditional_edges(
        "initialize",
        route_after_initialize,
        {
            "prompt_router": "prompt_router",
            "human_checkpoint": "human_checkpoint",
        }
    )

    # Add edges
    workflow.add_edge("prompt_router", "video_generation")

    # Conditional routing after video generation
    workflow.add_conditional_edges(
        "video_generation",
        route_after_video_generation,
        {
            "ai_review": "ai_review",
            "failed": END,
        }
    )

    # Conditional routing after AI review
    workflow.add_conditional_edges(
        "ai_review",
        route_after_ai_review,
        {
            "video_generation": "video_generation",
            "human_checkpoint": "human_checkpoint",
            "failed": END,
        }
    )

    # Conditional routing after human checkpoint
    workflow.add_conditional_edges(
        "human_checkpoint",
        route_after_human_checkpoint,
        {
            "video_generation": "video_generation",
            "end": END,
            "wait": END,
            "failed": END,
        }
    )

    # Compile the graph
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_phase3_pipeline(
    shot_id: str,
    show_id: str = "",
    image_version: Optional[str] = None,
    job_id: Optional[str] = None,
    human_decision: Optional[str] = None,
    human_updated_prompt: Optional[str] = None,
    human_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the Phase 3 video generation pipeline using LangGraph

    Args:
        shot_id: Shot ID to process
        show_id: Show ID for locating the shot in production_projects
        image_version: Optional version string of the shot image to use (e.g., "v0", "v1", "v2")
        job_id: Optional job ID for tracking in production_pipelines collection
        human_decision: Optional human decision (approved/needs_changes)
        human_updated_prompt: Optional human-updated prompt
        human_feedback: Optional human feedback/comments

    Returns:
        Final state dictionary with all results
    """
    logger.info("\n" + "🎉 "*40)
    logger.info("PHASE 3 LANGGRAPH PIPELINE STARTING")
    logger.info("🎉 "*40 + "\n")

    # Log the image version if specified
    if image_version is not None:
        logger.info(f"Using image version: {image_version}")

    # Initialize state
    initial_state: Phase3State = {
        "shot_id": shot_id,
        "show_id": show_id,
        "episode_number": 0,
        "shot_data": {},
        "image_version": image_version,
        "job_id": job_id,
        "generation_strategy": "",
        "video_prompt": "",
        "prompt_version": "",
        "start_image_url": "",
        "video_generation_task_id": None,
        "generated_video_url": None,
        "video_generation_status": "pending",
        "video_generation_attempt": 0,
        "max_video_generation_attempts": 3,
        "review_result": None,
        "review_decision": None,
        "review_score": None,
        "suggested_prompt": None,
        "suggested_prompt_reasoning": None,
        "ai_review_status": "pending",
        "human_decision": human_decision,
        "human_updated_prompt": human_updated_prompt,
        "human_feedback": human_feedback,
        "human_checkpoint_status": "pending",
        "human_regeneration_attempt": 0,
        "max_human_regeneration_attempts": 3,
        "current_node": "initialize",
        "pipeline_status": "running",
        "error_message": None,
        "mongodb_save_status": "pending",
        "video_versions": [],
        "current_version": 0,
    }

    # Create and run workflow
    app = create_phase3_workflow()

    # Execute the workflow
    final_state = app.invoke(initial_state)

    # Print results
    logger.info("\n" + "🎉 "*40)
    logger.info("PHASE 3 PIPELINE COMPLETED")
    logger.info("🎉 "*40)

    logger.info(f"\nPIPELINE STATUS: {final_state['pipeline_status']}")
    logger.info(f"📹 Generated video: {final_state.get('generated_video_url', 'N/A')}")
    logger.info(f"AI Review score: {final_state.get('review_score', 0)}/100")
    logger.info(f"Video versions generated: {len(final_state.get('video_versions', []))}")

    if final_state["pipeline_status"] == "failed":
        logger.error(f"\nERROR: {final_state.get('error_message', 'Unknown error')}")

    return final_state


if __name__ == "__main__":
    # Example usage
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    # Example: Process a specific shot
    shot_id = "S01E01_001"  # Replace with actual shot ID

    if len(sys.argv) > 1:
        shot_id = sys.argv[1]

    logger.info(f"Processing shot: {shot_id}")

    # Run pipeline
    result = run_phase3_pipeline(shot_id=shot_id)

    logger.info("\nPipeline execution complete!")
