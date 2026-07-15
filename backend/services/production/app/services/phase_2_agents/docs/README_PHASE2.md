# Phase 2 Agents - LangGraph Implementation

This directory contains the Phase 2 agents implemented with LangGraph workflow orchestration, following the same pattern as Phase 1 agents.

## Overview

Phase 2 implements a 3-agent pipeline with human approval checkpoint:

1. **Agent 1: Shot Strategy Agent** - Analyzes shot lists and determines generation strategies
2. **Human Approval Checkpoint** - Pauses for human review and approval of strategies  
3. **Agent 2: Image Prompt Generator Agent** - Generates cinematic image prompts for each shot
4. **Agent 3: Prompt Review Agent** - Reviews and refines prompts for visual continuity

## Flow

```
Shot List → Agent 1 (Strategy) → Human Approval → Agent 2 (Image Prompts) → Agent 3 (Prompt Review) → Final Output
```

## Key Files

- `workflow_state.py` - LangGraph state schema for Phase 2
- `langgraph_workflow.py` - Main workflow orchestration with LangGraph
- `agent_shot_strategy/` - Strategy analysis agent
- `image_prompt_generator_agent.py` - Image prompt generation agent
- `agent_prompt_review/` - Prompt review and refinement agent

## API Endpoints

The Phase 2 workflow is exposed through the following endpoints:

### `/api/v1/phase2/run-strategy-agent`
- Runs only the strategy agent
- Analyzes shot list and saves strategies to MongoDB
- Returns job_id for tracking

### `/api/v1/phase2/approve-strategy`
- Human approval endpoint for strategies
- Updates strategy_approval field in MongoDB
- Automatically runs image agents after approval (default behavior)


### `/api/v1/phase2/status/{job_id}`
- Get current status of a workflow job
- Returns agent statuses and pipeline progress

### `/api/v1/phase2/results/{job_id}`
- Get complete results of a workflow job
- Returns all agent outputs and generated files

## Usage Example

```python
# 1. Run strategy agent
POST /api/v1/phase2/run-strategy-agent
{
    "shot_list_request": {
        "episode_id": "E01",
        "title": "Episode 1",
        "shots": [...]
    },
    "show_id": "my_show",
    "episode_number": 1,
    "scene_description": "Scene description"
}

# 2. Approve strategies (human action)
POST /api/v1/phase2/approve-strategy
{
    "show_id": "my_show",
    "episode_number": 1,
    "approval_status": true,
    "feedback": {...}
}

# 3. Check status (image agents run automatically after approval)
GET /api/v1/phase2/status/{job_id}

# 4. Get results
GET /api/v1/phase2/results/{job_id}
```

## Human Approval Flow

The workflow includes a human approval checkpoint after the strategy agent:

1. Strategy agent analyzes shots and saves to MongoDB
2. Workflow pauses and waits for human approval
3. Human reviews strategies via `/approve-strategy` endpoint
4. If approved, image agents run automatically (Agent 2 & 3)
5. If rejected, workflow ends

**Note:** Image agents (Agent 2 & 3) now run automatically after strategy approval, eliminating the need for a separate `/run-image-agents` endpoint.

## MongoDB Integration

The workflow integrates with MongoDB Atlas:

- Saves strategy analysis results
- Stores generated image prompts
- Updates with prompt review results
- Tracks approval status

## Testing

Run the test script to verify the implementation:

```bash
cd backend/services/production/app/services/phase_2_agents
python test_phase2_workflow.py
```

## Environment Variables

Required environment variables:

- `GOOGLE_API_KEY` - For Gemini API access
- `MONGODB_ATLAS_URI` - MongoDB connection string
- `MONGODB_DATABASE_NAME` - Database name (default: "production")
- `MONGODB_SHOTS_COLLECTION` - Collection name (default: "shots")

## Differences from Old IPV Implementation

1. **LangGraph Integration** - Uses LangGraph for workflow orchestration
2. **Unified State Management** - All agents share a common state schema
3. **Human Approval Checkpoint** - Built-in human approval flow
4. **MongoDB Integration** - Direct MongoDB operations within agents
5. **Background Job Processing** - Async job processing with status tracking
6. **Error Handling** - Comprehensive error handling and recovery

## Migration from Old IPV

The old IPV endpoints are preserved and work the same way:

- `/run-strategy-agent` → `/api/v1/phase2/run-strategy-agent`
- `/approve-strategy` → `/api/v1/phase2/approve-strategy` (now auto-runs image agents)

**Note:** The `/run-image-agents` endpoint has been removed as image agents now run automatically after strategy approval.

The core functionality remains the same, but now with LangGraph orchestration, improved error handling, and automatic flow after approval.
