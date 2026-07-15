# LangGraph Phase 1 Pipeline

## Overview

The Phase 1 agents have been integrated with **LangGraph** for state management and workflow orchestration. This provides:

- ✅ **Centralized State Management** - Single source of truth across all agents
- ✅ **Automatic Error Handling** - Built-in retry logic and failure recovery
- ✅ **Conditional Routing** - Smart routing based on agent outputs
- ✅ **Clean Code** - Reduced print statements, focused logic
- ✅ **Easy Testing** - Simple test runner for the entire pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
│                                                              │
│  Script Input                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────┐│
│  │ Agent 1 │─────▶│ Agent 2 │─────▶│ Agent 3 │─────▶│ Ag4 ││
│  │ Extract │      │ Review  │      │ Prompts │      │ Opt ││
│  └─────────┘      └─────────┘      └─────────┘      └─────┘│
│       │                 │                │               │   │
│       └─────────────────┴────────────────┴───────────────┘   │
│                         Shared State                         │
└─────────────────────────────────────────────────────────────┘
```

## Files

### Core Files

1. **workflow_state.py** - Defines the shared state schema (TypedDict)
2. **langgraph_workflow.py** - Main workflow orchestrator with node definitions
3. **test_langgraph_pipeline.py** - Simple test runner

### Agent Files (Refactored)

- `agent_1_asset_generator.py` - Cleaned up, LangGraph-compatible
- `agent_2_asset_reviewer.py` - Cleaned up, LangGraph-compatible
- `agent_3_prompt_generator.py` - Cleaned up, LangGraph-compatible
- `agent_4_prompt_optimizer.py` - Cleaned up, LangGraph-compatible

## Usage

### Basic Usage

```python
from langgraph_workflow import run_phase1_pipeline

# Run the entire pipeline
result = run_phase1_pipeline(script_path="your_script.txt")

# Access results
print(f"Status: {result['pipeline_status']}")
print(f"Final prompts: {result['optimized_prompts']}")
```

### With Human Feedback

```python
result = run_phase1_pipeline(
    script_path="script.txt",
    agent1_feedback={
        "feedback_type": "modify",
        "missing_assets": {...}
    },
    agent2_feedback={
        "approve_enhancements": True,
        "approve_missing_additions": {...}
    }
)
```

### Run Test

```bash
cd backend/services/production/app/services/phase_1_agents
python test_langgraph_pipeline.py
```

## State Flow

The `Phase1State` TypedDict tracks:

```python
{
    # Input
    "script_path": str,
    "script_content": str,

    # Agent 1 outputs
    "extracted_assets": {...},
    "agent1_status": "completed",

    # Agent 2 outputs
    "review_results": {...},
    "enhanced_assets": {...},
    "agent2_status": "completed",

    # Agent 3 outputs
    "generated_prompts": {...},
    "agent3_status": "completed",

    # Agent 4 outputs
    "optimized_prompts": {...},
    "agent4_status": "completed",

    # Pipeline tracking
    "current_agent": "agent_2",
    "pipeline_status": "running",
    "output_files": [...],
}
```

## Benefits Over Manual JSON Passing

### Before (Manual)
```python
# Run Agent 1
agent1 = AssetGeneratorAgent(api_key)
agent1.load_script(script_path)
agent1.extract_assets()
agent1_output = agent1.save_asset_records()

# Run Agent 2 (manually load Agent 1 output)
agent2 = AssetReviewerAgent(api_key)
agent2.load_agent1_output(agent1_output)  # Manual file loading
agent2.review_assets()
# ... repeat for all agents
```

### After (LangGraph)
```python
# Run entire pipeline with state management
result = run_phase1_pipeline(script_path="script.txt")
# State automatically passed between all agents
```

## Conditional Routing

The workflow includes conditional edges for smart routing:

```python
workflow.add_conditional_edges(
    "agent_1",
    should_continue,
    {
        "agent_2": "agent_2",    # Success → next agent
        "failed": END,           # Failure → end
    }
)
```

## Error Handling

Each node includes error handling:

```python
try:
    # Agent logic
    result = agent.process()
    return {**state, "status": "completed"}
except Exception as e:
    return {
        **state,
        "status": "failed",
        "error_message": str(e)
    }
```

## Human-in-the-Loop

Human feedback can be injected at any stage:

```python
result = run_phase1_pipeline(
    script_path="script.txt",
    agent1_feedback={"feedback_type": "modify", ...},
    agent2_feedback={"approve_enhancements": True, ...},
    agent3_feedback={"approve_all": True},
    agent4_feedback={"approve_all": True}
)
```

## Next Steps

To extend to remaining agents (5-8):

1. Add agent nodes to `langgraph_workflow.py`
2. Update `workflow_state.py` with new state fields
3. Add conditional routing for each new agent
4. Update test file

## Dependencies

```bash
pip install langgraph typing-extensions
```

## Comparison: Before vs After

| Feature | Before | After (LangGraph) |
|---------|--------|-------------------|
| State Management | Manual JSON files | Centralized TypedDict |
| Agent Communication | File I/O | Direct state passing |
| Error Recovery | Manual try/catch | Built-in retry logic |
| Conditional Logic | Manual if/else | Conditional edges |
| Testing | Run each agent separately | Single test runner |
| Code Cleanliness | Many print statements | Minimal logging |
| Observability | Print debugging | State inspection |
| Scalability | Hard to add agents | Easy to extend graph |

---

**Generated with LangGraph integration for Phase 1 Agents**
