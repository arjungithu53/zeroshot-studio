"""
Pipeline Service Layer
======================
Handles CRUD operations for pipeline jobs in MongoDB.
Pipelines are lightweight job trackers that reference projects.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pymongo.collection import Collection

from app.config import get_pipelines_collection

logger = logging.getLogger(__name__)
from app.models.mongodb.pipelines import (
    PipelineJobModel,
    PipelineJobCreate,
    PipelineJobResponse
)


class PipelineService:
    """Service layer for pipeline job tracking"""

    def __init__(self) -> None:
        self.client, self.collection = get_pipelines_collection()

    def create_job(self, job_data) -> Dict[str, str]:
        """
        Create a new pipeline job

        Args:
            job_data: Job creation data (PipelineJobCreate model or dict)
                     Can include: project_id, movie_id, assets_collection_id, shot_id, type, etc.

        Returns:
            Dict with job_id and success status
        """
        job_id = str(uuid.uuid4())

        # Handle both Pydantic models and dicts
        if isinstance(job_data, dict):
            # Dict-based creation (for movie workflow)
            job_dict = {
                "job_id": job_id,
                "status": job_data.get("status", "pending"),
                "pipeline_status": job_data.get("pipeline_status", "pending"),
                "current_agent": job_data.get("current_agent", "agent_1"),
                "created_at": job_data.get("created_at", datetime.utcnow()),
                "updated_at": datetime.utcnow()
            }

            # Add optional fields if present
            if "project_id" in job_data:
                job_dict["project_id"] = job_data["project_id"]
            if "movie_id" in job_data:
                job_dict["movie_id"] = job_data["movie_id"]
            if "assets_collection_id" in job_data:
                job_dict["assets_collection_id"] = job_data["assets_collection_id"]
            if "shot_id" in job_data:
                job_dict["shot_id"] = job_data["shot_id"]
            if "type" in job_data:
                job_dict["type"] = job_data["type"]
            if "max_regenerations" in job_data:
                job_dict["max_regenerations"] = job_data["max_regenerations"]
        else:
            # Pydantic model-based creation (legacy workflow)
            job_dict = {
                "job_id": job_id,
                "project_id": job_data.project_id,
                "shot_id": job_data.shot_id,
                "max_regenerations": job_data.max_regenerations,
                "status": "pending",
                "pipeline_status": "pending",
                "current_agent": "agent_1",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

        # Insert into MongoDB
        self.collection.insert_one(job_dict)

        return {"job_id": job_id, "success": True}

    def get_job(self, job_id: str) -> Optional[PipelineJobModel]:
        """
        Get job by ID

        Args:
            job_id: Job identifier

        Returns:
            Job model or None if not found
        """
        job_data = self.collection.find_one({"job_id": job_id})
        if job_data:
            return PipelineJobModel(**job_data)
        return None

    def get_jobs(self, job_ids: List[str]) -> Dict[str, PipelineJobModel]:
        """
        Fetch multiple jobs by ID in a single query.

        Args:
            job_ids: List of job identifiers

        Returns:
            Dict mapping job_id -> PipelineJobModel for found jobs
        """
        if not job_ids:
            return {}
        job_docs = self.collection.find({"job_id": {"$in": job_ids}})
        return {doc["job_id"]: PipelineJobModel(**doc) for doc in job_docs}

    def get_job_by_project_id(self, project_id: str) -> Optional[PipelineJobModel]:
        """
        Get the most recent job for a project

        Args:
            project_id: Project identifier

        Returns:
            Most recent job model or None if not found
        """
        job_data = self.collection.find_one(
            {"project_id": project_id},
            sort=[("created_at", -1)]  # Get most recent job
        )
        if job_data:
            return PipelineJobModel(**job_data)
        return None

    def get_job_by_movie_id(self, movie_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent job for a movie

        Args:
            movie_id: Movie identifier

        Returns:
            Most recent job dict or None if not found
        """
        job_data = self.collection.find_one(
            {"movie_id": movie_id},
            sort=[("created_at", -1)]  # Get most recent job
        )
        return job_data

    def update_job_status(self, job_id: str, status: str, **kwargs) -> bool:
        """
        Update job status and other fields

        Args:
            job_id: Job identifier
            status: New status
            **kwargs: Additional fields to update (can include celery_task_id)

        Returns:
            True if updated, False if not found
        """
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow(),
            **kwargs
        }

        result = self.collection.update_one(
            {"job_id": job_id},
            {"$set": update_data}
        )

        return result.modified_count > 0

    def update_job_current_agent(self, job_id: str, current_agent: str) -> bool:
        """
        Update the current_agent field on a pipeline job.

        Called by every Phase 4 agent on completion to advance the tracker.

        Args:
            job_id:        Job identifier.
            current_agent: Name of the agent now running (e.g. "agent_2").

        Returns:
            True if updated, False if not found.
        """
        result = self.collection.update_one(
            {"job_id": job_id},
            {"$set": {"current_agent": current_agent, "updated_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    def update_job_celery_task_id(self, job_id: str, celery_task_id: str) -> bool:
        """
        Update job with Celery task ID

        This links the job to a Celery task for status tracking.
        Called immediately after dispatching a Celery task.

        Args:
            job_id: Job identifier
            celery_task_id: Celery task ID from task.apply_async()

        Returns:
            True if updated, False if not found

        Why separate method?
        -------------------
        - Task ID is assigned AFTER job creation
        - Allows atomic update without race conditions
        - Clear intent in code when linking job to task
        """
        update_data = {
            "celery_task_id": celery_task_id,
            "updated_at": datetime.utcnow()
        }

        result = self.collection.update_one(
            {"job_id": job_id},
            {"$set": update_data}
        )

        return result.modified_count > 0

    def update_job_state(self, job_id: str, state_data: Dict[str, Any]) -> bool:
        """
        Update job with workflow state data (only tracking/status fields)

        Args:
            job_id: Job identifier
            state_data: State dictionary from workflow

        Returns:
            True if updated, False if not found
        """
        # Extract only tracking fields (NO asset data)
        update_data = {
            "updated_at": datetime.utcnow(),
            "current_agent": state_data.get("current_agent", ""),
            "pipeline_status": state_data.get("pipeline_status", ""),
            "agent1_status": state_data.get("agent1_status", "pending"),
            "agent2_status": state_data.get("agent2_status", "pending"),
            "agent3_status": state_data.get("agent3_status", "pending"),
            "agent4_status": state_data.get("agent4_status", "pending"),
            "agent5_status": state_data.get("agent5_status", "pending"),
            "agent6_status": state_data.get("agent6_status", "pending"),
            "agent7_status": state_data.get("agent7_status", "pending"),
            "agent8_status": state_data.get("agent8_status", "pending"),
            # Phase 3 agent statuses
            "agent17_status": state_data.get("agent17_status", "pending"),
            "agent18_status": state_data.get("agent18_status", "pending"),
            "agent19_status": state_data.get("agent19_status", "pending"),
            "phase_3_checkpoint_status": state_data.get("phase_3_checkpoint_status", "pending"),
            "regeneration_count": state_data.get("regeneration_count", 0),
            "output_files": state_data.get("output_files", []),
            "error_message": state_data.get("error_message"),
        }

        def _serialize_state_value(value: Any) -> Any:
            """Convert workflow state values into Mongo-safe primitives."""
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if isinstance(value, list):
                return [_serialize_state_value(v) for v in value]
            if isinstance(value, dict):
                return {k: _serialize_state_value(v) for k, v in value.items()}
            return value

        resume_fields = [
            "shot_list_request",
            "annotated_shot_list",
            "strategy_analysis_results",
            "image_prompts_generated",
            "reviewed_prompts",
            "shot_designs",
            "modified_prompts",
            "prompt_approval_decision",
            "prompt_approval_feedback",
            "generated_images",
            "image_reviews",
            "edited_shots",
            "edit_loop_iterations",
            "shots_needing_edit",
            "shots_edit_instructions",
            "shots_needing_regeneration",
            "shots_approved",
            "shots_max_retries",
            "regenerate_loop_iterations",
            "product_image_url",
            "v1_project_id",
            "product_review_results",
            "product_review_iterations",
            "product_fix_prompts",
            "product_corrected_images",
            "shots_needing_product_fix",
            "shots_product_approved",
            "final_approval_decision",
            "final_approval_feedback",
        ]

        resume_state = {
            "show_id": state_data.get("show_id", ""),
            "episode_number": state_data.get("episode_number", 0),
            "project_id": state_data.get("project_id", ""),
            "movie_id": state_data.get("movie_id", ""),  # CRITICAL: needed for visual_style fetch
            "job_id": state_data.get("job_id", ""),  # CRITICAL: needed for tracking
            "scene_description": state_data.get("scene_description", ""),
            "current_agent": state_data.get("current_agent", ""),
            "pipeline_status": state_data.get("pipeline_status", ""),
            "agent1_status": state_data.get("agent1_status", "pending"),
            "agent2_status": state_data.get("agent2_status", "pending"),
            "agent3_status": state_data.get("agent3_status", "pending"),
            "agent12_status": state_data.get("agent12_status", "pending"),
            "agent13_status": state_data.get("agent13_status", "pending"),
            "agent14_status": state_data.get("agent14_status", "pending"),
            "agent15_status": state_data.get("agent15_status", "pending"),
            "agent15A_status": state_data.get("agent15A_status", "pending"),
            "agent16_status": state_data.get("agent16_status", "pending"),
            "agent17_status": state_data.get("agent17_status", "pending"),
            "agent18_status": state_data.get("agent18_status", "pending"),
            "agent7_status": state_data.get("agent7_status", "pending"),
        }

        for field in resume_fields:
            if field in state_data:
                resume_state[field] = _serialize_state_value(state_data[field])

        # DEBUG: Print critical fields before saving (for testing)
        print(f"[DEBUG] resume_state before save: movie_id={resume_state.get('movie_id')}, job_id={resume_state.get('job_id')}")
        logger.info(f"[update_job_state] Saving state with movie_id={resume_state.get('movie_id')}, job_id={resume_state.get('job_id')}")

        update_data["state"] = resume_state

        # Update status based on pipeline_status
        if state_data.get("pipeline_status") == "completed":
            update_data["status"] = "completed"
            update_data["completed_at"] = datetime.utcnow()
        elif state_data.get("pipeline_status") == "failed":
            update_data["status"] = "failed"
        elif state_data.get("pipeline_status") in ["waiting_for_human_approval", "waiting_for_approval", "waiting_for_prompt_approval", "waiting_for_final_approval", "phase_3_checkpoint"]:
            update_data["status"] = "waiting_for_human_approval"
        elif state_data.get("pipeline_status") == "waiting_for_human":
            # Phase 3 specific checkpoint status
            update_data["status"] = "phase_3_checkpoint"
            update_data["phase_3_checkpoint_status"] = "waiting"
        elif state_data.get("pipeline_status") in ["running", "generating_variations", "regenerating_images"]:
            update_data["status"] = "running"

        result = self.collection.update_one(
            {"job_id": job_id},
            {"$set": update_data}
        )

        return result.modified_count > 0

    def set_human_approval(self, job_id: str, decision: str, feedback: Optional[Dict] = None) -> bool:
        """
        Set human approval decision

        Args:
            job_id: Job identifier
            decision: 'approve' or 'edit_prompts'
            feedback: Optional feedback data

        Returns:
            True if updated, False if not found
        """
        update_data = {
            "human_approval_decision": decision,
            "human_approval_feedback": feedback,
            "updated_at": datetime.utcnow()
        }

        result = self.collection.update_one(
            {"job_id": job_id},
            {"$set": update_data}
        )

        return result.modified_count > 0

    def update_approved_assets(self, job_id: str, approved_assets_list: List[str], feedback: Optional[Dict] = None) -> bool:
        """
        Update the list of approved assets at checkpoint

        Args:
            job_id: Job identifier
            approved_assets_list: List of approved asset IDs
            feedback: Optional feedback data for these approvals

        Returns:
            True if updated, False if not found
        """
        update_data = {
            "approved_assets_list": approved_assets_list,
            "updated_at": datetime.utcnow()
        }

        # Optionally update feedback if provided
        if feedback:
            update_data["human_approval_feedback"] = feedback

        result = self.collection.update_one(
            {"job_id": job_id},
            {"$set": update_data}
        )

        return result.modified_count > 0

    def mark_checkpoint_finalized(self, job_id: str) -> bool:
        """
        Mark the checkpoint as finalized (user clicked Continue)

        Args:
            job_id: Job identifier

        Returns:
            True if updated, False if not found
        """
        result = self.collection.update_one(
            {"job_id": job_id},
            {"$set": {
                "checkpoint_approved": True,
                "updated_at": datetime.utcnow()
            }}
        )
        return result.modified_count > 0

    def to_response(self, job_model: PipelineJobModel) -> PipelineJobResponse:
        """
        Convert job model to response format

        Args:
            job_model: Job model

        Returns:
            Job response with celery_task_id for monitoring

        Why include celery_task_id?
        ---------------------------
        - Frontend can poll /task-status/{celery_task_id} for real-time progress
        - Shows current agent, completion percentage, errors
        - Better UX than polling job status alone
        """
        # Reconstruct approved_assets from human_approval_feedback
        approved_assets = []
        if job_model.human_approval_feedback:
            approved_assets = job_model.human_approval_feedback.get("approved_assets", [])

        return PipelineJobResponse(
            job_id=job_model.job_id,
            project_id=job_model.project_id,
            movie_id=job_model.movie_id,
            assets_collection_id=job_model.assets_collection_id,
            shot_id=job_model.shot_id,
            type=job_model.type,
            status=job_model.status,
            current_agent=job_model.current_agent,
            pipeline_status=job_model.pipeline_status,
            created_at=job_model.created_at,
            updated_at=job_model.updated_at,
            celery_task_id=job_model.celery_task_id,  # ← Celery task ID for monitoring
            agent_statuses={
                "agent_1": job_model.agent1_status,
                "agent_2": job_model.agent2_status,
                "agent_3": job_model.agent3_status,
                "agent_4": job_model.agent4_status,
                "agent_5": job_model.agent5_status,
                "agent_6": job_model.agent6_status,
                "agent_7": job_model.agent7_status,
                "agent_8": job_model.agent8_status,
                "agent_17": job_model.agent17_status,
                "agent_18": job_model.agent18_status,
                "agent_19": job_model.agent19_status,
                "phase_3_checkpoint": job_model.phase_3_checkpoint_status,
            },
            waiting_for_approval=(job_model.status == "waiting_for_human_approval"),
            regeneration_count=job_model.regeneration_count,
            checkpoint_approved=job_model.checkpoint_approved,
            approved_assets=approved_assets,
            output_files=job_model.output_files,
            error_message=job_model.error_message
        )
