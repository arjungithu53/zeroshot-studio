"""
Celery Tasks Package
===================

This package contains all Celery task definitions for the production service.

Task Organization:
-----------------
- phase1_tasks.py: Phase 1 workflow (8-agent asset generation pipeline)
- phase2_tasks.py: Phase 2 workflow (shot design and image generation)
- phase3_tasks.py: Phase 3 workflow (video generation)

Why separate task files?
------------------------
- Better organization and maintainability
- Different queues for different resource requirements
- Easier to scale specific workflows independently

Current Status:
--------------
Phase 1: ✅ Implemented with Celery + SQS
Phase 2: ✅ Implemented with Celery + SQS
Phase 3: ✅ Implemented with Celery + SQS
"""

from app.tasks.phase1_tasks import (
    run_phase1_workflow_task,
    resume_phase1_workflow_task,
    retry_failed_asset_task,
)

from app.tasks.phase2_tasks import (
    run_phase2_workflow_task,
    resume_phase2_after_strategy_approval_task,
    resume_phase2_after_prompt_approval_task,
    resume_phase2_after_final_approval_task,
)

from app.tasks.phase3_tasks import (
    run_phase3_workflow_task,
    resume_phase3_workflow_task,
)

__all__ = [
    # Phase 1 tasks
    'run_phase1_workflow_task',
    'resume_phase1_workflow_task',
    'retry_failed_asset_task',
    # Phase 2 tasks
    'run_phase2_workflow_task',
    'resume_phase2_after_strategy_approval_task',
    'resume_phase2_after_prompt_approval_task',
    'resume_phase2_after_final_approval_task',
    # Phase 3 tasks
    'run_phase3_workflow_task',
    'resume_phase3_workflow_task',
]
