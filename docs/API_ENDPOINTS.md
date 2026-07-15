# API Endpoints Overview

## Projects

### `GET /api/v1/projects` (List Projects)
Retrieves a list of all existing projects within the system. 
This is typically used to populate project dashboards and supports pagination or basic filtering.

### `POST /api/v1/projects/create` (Create Project)
Initializes a new, empty project record in the database. 
Returns the unique ID and default metadata for the newly created project.

### `POST /api/v1/projects/create-name` (Create Project Name)
Assigns or updates the human-readable name for a project, often used during the initial setup wizard. 
Keeps project creation flexible by separating name assignment from instantiation.

### `POST /api/v1/projects/{project_id}/upload-files` (Upload Script And Shotlist)
Allows the user to attach essential raw files (like the central script and shotlist csv) to a specific project. 
These files serve as the foundational input for the video generation pipelines.

### `GET /api/v1/projects/{project_id}` (Get Project)
Fetches the complete, detailed metadata and configuration for a specific project using its unique identifier. 
Used to load the detailed view of a single project.

### `GET /api/v1/projects/{project_id}/status` (Get Project Status)
Returns the high-level life-cycle state or overall completion percentage of a project. 
Useful for tracking whether a project is in pre-production, processing, or completed.

---

## Phase 1 Workflow

### `POST /api/v1/phase1/start` (Start Workflow)
Triggers the execution of the Phase 1 generation pipeline for a project. 
Initializes backend tasks (like Celery jobs) and returns a `job_id` for tracking progress.

### `POST /api/v1/phase1/upload-script` (Upload Script)
Handles the targeted uploading and preliminary validation of a script file specifically for Phase 1 processing. 
May trigger immediate parsing algorithms to unblock the rest of the workflow.

### `GET /api/v1/phase1/status/{job_id}` (Get Job Status)
Retrieves the execution state (e.g., running, completed, pending, failed) of a specific Phase 1 asynchronous job. 
Used constantly by the frontend to show progress bars to the user.

### `GET /api/v1/phase1/results/{job_id}` (Get Job Results)
Provides the final structured data, intermediate outputs, or artifacts produced by a recently run Phase 1 job. 
Called once the job status returns as completed.

### `GET /api/v1/phase1/results/by-project/{project_id}` (Get Results By Project)
Aggregates and returns the historical output data from all Phase 1 jobs associated with a specific project. 
Useful for compiling a historical view or final project summary.

### `POST /api/v1/phase1/checkpoint/approve/{job_id}` (Approve Checkpoint)
Signals that a human reviewer has accepted the AI outputs at a specific generation checkpoint. 
This action unblocks the workflow, allowing it to advance to the next set of automated tasks.

### `POST /api/v1/phase1/checkpoint/finalize/{job_id}` (Finalize Checkpoint)
Locks in the approved assets at a checkpoint, marking them as immutable. 
Triggers the final compilation or transition to Phase 2 of the production pipeline.

### `POST /api/v1/phase1/checkpoint/edit-prompt/{job_id}` (Edit Asset Prompt)
Allows a user to manually modify the underlying AI generation prompt for an asset that fell short of expectations. 
Submitting this usually queues a targeted regeneration of that specific asset.

### `GET /api/v1/phase1/outputs/{job_id}` (Get Output Files)
Yields the direct access URLs or download links for the final generated media/document files from a specific job. 
Enables users to download their final deliverables.

### `GET /api/v1/phase1/failed-assets/{job_id}` (Get Failed Assets)
Identifies and lists any specific subset of assets that threw errors or timed out during the job's execution. 
Provides helpful error details so the user knows what needs manual intervention.

### `POST /api/v1/phase1/retry-asset/{job_id}` (Retry Failed Asset)
Re-submits a specific failed asset's generation task back to the processing queue. 
Saves time by only reprocessing broken items instead of restarting the entire job.

### `GET /api/v1/phase1/task-status/{task_id}` (Get Task Status)
Checks the granular, low-level status of an individual background worker task (like an AWS SQS or Celery task). 
Often used for deep backend debugging or micro-progress tracking.

### `POST /api/v1/phase1/cancel-task/{task_id}` (Cancel Task)
Terminates a pending or currently executing background task to free up worker resources. 
Useful if the user realizes they made a mistake in their prompt and wants to abort.

### `GET /api/v1/phase1/task-info/{job_id}` (Get Task Info)
Retrieves comprehensive metadata about the tasks making up a job, including timestamps, worker nodes, and queue times. 
Primarily used for internal analytics, performance monitoring, or advanced debugging.
