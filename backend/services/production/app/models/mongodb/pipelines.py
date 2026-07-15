"""
MongoDB Models for Pipeline Job Tracking
=========================================
Stores lightweight job execution state for tracking pipeline runs.
All actual asset data is stored in production_projects and production_assets collections.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2"""
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema
        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ])
        ], serialization=core_schema.plain_serializer_function_ser_schema(str))

    @classmethod
    def validate(cls, v: Any) -> ObjectId:
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


class PipelineJobModel(BaseModel):
    """MongoDB model for pipeline job tracking (lightweight)"""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    job_id: str = Field(..., description="Unique job identifier (UUID)")

    # Reference to actual data
    project_id: Optional[str] = Field(None, description="Reference to production_projects collection")
    movie_id: Optional[str] = Field(None, description="Reference to movies collection (for movie workflows)")
    assets_collection_id: Optional[str] = Field(None, description="Reference to assets_collection (for movie workflows)")
    shot_id: Optional[str] = Field(None, description="Shot ID for Phase 3 video generation jobs")
    type: Optional[str] = Field(None, description="Job type: phase1_project, phase1_movie, phase2, phase3")

    # Job metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Celery task tracking
    celery_task_id: Optional[str] = Field(
        None,
        description="Celery task ID for tracking async execution. "
                   "Used to query task status, progress, and results from Celery/SQS."
    )

    # Job status tracking
    status: str = Field(
        default="pending",
        description="Job status: pending, running, waiting_for_human_approval, completed, failed"
    )
    current_agent: str = Field(default="agent_1", description="Current agent in workflow")
    pipeline_status: str = Field(default="pending", description="Overall pipeline status")

    # Agent execution statuses (just status, no data)
    agent1_status: str = Field(default="pending")
    agent2_status: str = Field(default="pending")
    agent3_status: str = Field(default="pending")
    agent4_status: str = Field(default="pending")
    agent5_status: str = Field(default="pending")
    agent6_status: str = Field(default="pending")
    agent7_status: str = Field(default="pending")
    agent8_status: str = Field(default="pending")

    # Phase 3 agent statuses (17-19 for video generation pipeline)
    agent17_status: str = Field(default="pending", description="Video prompt generation agent")
    agent18_status: str = Field(default="pending", description="Video generation agent")
    agent19_status: str = Field(default="pending", description="AI video review agent")
    phase_3_checkpoint_status: str = Field(default="pending", description="Human checkpoint for video approval")

    # Human checkpoint tracking
    human_approval_decision: Optional[str] = None
    human_approval_feedback: Optional[Dict[str, Any]] = None
    regeneration_count: int = Field(default=0)
    max_regenerations: int = Field(default=5)
    approved_assets_list: List[str] = Field(default_factory=list, description="List of approved asset IDs during checkpoint")
    checkpoint_approved: bool = Field(default=False, description="Whether the checkpoint has been finalized")

    # Error handling
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Optional: track output file locations for convenience
    output_files: List[str] = Field(default_factory=list)

    # Minimal state for resume operations (only essential fields, no large data)
    state: Optional[Dict[str, Any]] = Field(default=None, description="Minimal workflow state for resume operations")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "project_id": "507f1f77bcf86cd799439011",
                "status": "running",
                "current_agent": "agent_3",
                "pipeline_status": "running",
                "agent1_status": "completed",
                "agent2_status": "completed",
                "agent3_status": "in_progress"
            }
        }


class PipelineJobCreate(BaseModel):
    """Request model for creating a new pipeline job"""
    project_id: str = Field(..., description="Reference to production_projects collection")
    shot_id: Optional[str] = Field(None, description="Shot ID for Phase 3 video generation jobs")
    max_regenerations: int = Field(default=5, description="Max image regeneration attempts")


class PipelineJobResponse(BaseModel):
    """Response model for pipeline job"""
    job_id: str
    project_id: Optional[str] = None
    movie_id: Optional[str] = None
    assets_collection_id: Optional[str] = None
    shot_id: Optional[str] = None
    type: Optional[str] = None
    status: str
    current_agent: str
    pipeline_status: str
    created_at: datetime
    updated_at: datetime

    # Celery task tracking
    celery_task_id: Optional[str] = Field(
        None,
        description="Celery task ID for monitoring task progress via /api/v1/phase1/task-status/{task_id}"
    )

    # Agent statuses
    agent_statuses: Dict[str, str] = Field(default_factory=dict)

    # Human checkpoint info
    waiting_for_approval: bool = False
    regeneration_count: int = 0
    checkpoint_approved: bool = False
    approved_assets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of approved assets with their feedback"
    )

    # Output
    output_files: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "project_id": "507f1f77bcf86cd799439011",
                "status": "waiting_for_human_approval",
                "current_agent": "human_checkpoint",
                "pipeline_status": "waiting_for_human_approval",
                "agent_statuses": {
                    "agent_1": "completed",
                    "agent_2": "completed",
                    "agent_3": "completed",
                    "agent_4": "completed",
                    "agent_5": "completed",
                    "agent_6": "completed"
                },
                "waiting_for_approval": True,
                "regeneration_count": 0,
                "output_files": [
                    "output/agent1_asset_records_20251016.json",
                    "output/agent6_review_results_20251016.json"
                ]
            }
        }


class AssetApproval(BaseModel):
    """Individual asset approval with feedback"""
    asset_id: str = Field(..., description="UUID of the asset")
    asset_type: str = Field(..., description="Type: 'character', 'location', or 'prop'")
    feedback: Optional[str] = Field(None, description="Feedback for this specific asset")

    class Config:
        json_schema_extra = {
            "example": {
                "asset_id": "550e8400-e29b-41d4-a716-446655440000",
                "asset_type": "character",
                "feedback": "Looks great!"
            }
        }


class HumanApprovalRequest(BaseModel):
    """Request model for human approval - user approves specific assets with individual feedback"""
    approved_assets: List[AssetApproval] = Field(
        ...,
        description="List of approved assets with individual feedback"
    )
    global_feedback: Optional[str] = Field(None, description="Optional general feedback about all assets")

    class Config:
        json_schema_extra = {
            "example": {
                "approved_assets": [
                    {
                        "asset_id": "550e8400-e29b-41d4-a716-446655440000",
                        "asset_type": "character",
                        "feedback": "Perfect character design!"
                    },
                    {
                        "asset_id": "550e8400-e29b-41d4-a716-446655440001",
                        "asset_type": "location",
                        "feedback": "Good lighting and composition"
                    }
                ],
                "global_feedback": "Overall looking great!"
            }
        }


class AssetPromptEdit(BaseModel):
    """Request model for editing a specific asset's prompt and re-running Agent 7"""
    asset_id: str = Field(..., description="UUID of the asset to edit")
    asset_type: str = Field(..., description="Type: 'character', 'location', or 'prop'")
    edited_prompt: str = Field(..., description="The new/modified prompt for this asset")
    feedback: Optional[str] = Field(None, description="Optional feedback about why the edit was needed")

    class Config:
        json_schema_extra = {
            "example": {
                "asset_id": "550e8400-e29b-41d4-a716-446655440000",
                "asset_type": "character",
                "edited_prompt": "A young woman with curly red hair, wearing a blue dress, standing in a garden with bright sunlight",
                "feedback": "Original was too dark, needed more vibrant colors"
            }
        }
