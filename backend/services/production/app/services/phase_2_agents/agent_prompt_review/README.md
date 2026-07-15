# Agent 3: Prompt Review Agent

## Overview

The **Prompt Review Agent** is the third AI agent in Phase 2 of the project. It reviews and refines AI-generated image prompts to ensure **visual and narrative continuity** across shot sequences.

## Purpose

After Agent 2 (Prompt Generation Agent) creates initial image prompts (`prompt_image_draft`) for each shot, Agent 3:
1. Analyzes all prompts globally for overall consistency
2. Reviews consecutive shots pairwise for micro-continuity
3. Identifies and fixes continuity errors
4. Generates refined prompts (`prompt_image_reviewed`)
5. Saves detailed review reports locally
6. Updates MongoDB with the reviewed prompts

## Integration

This agent is **automatically integrated** into the `/analyze-shots-mongodb` flow:

```
/analyze-shots-mongodb endpoint:
  ├── Phase 1: Strategy Agent (Agent 1) → Determines generation strategies
  ├── Phase 2: Prompt Generation Agent (Agent 2) → Creates prompt_image_draft
  └── Phase 3: Prompt Review Agent (Agent 3) → Creates prompt_image_reviewed
```

## What It Checks

The agent reviews prompts for:

### Visual Consistency
- Lighting (direction, quality, color temperature, intensity)
- Weather conditions (fog, rain, sun, clouds)
- Time of day (morning, afternoon, sunset, night)
- Color palette and grading
- Atmospheric effects

### Spatial Continuity (Blocking)
- Character positions relative to each other
- Props and set pieces locations
- Screen direction consistency
- Spatial relationships

### Directional Consistency
- Sun position and light direction
- Character eye-lines and facing directions
- Camera angles and spatial logic

### Character Consistency
- Costume and appearance
- Emotional state
- Physical positioning

### Scene Detail Continuity
- Background elements
- Props and set dressing
- Established visual details

## Output

### MongoDB Field
- **`prompt_image_reviewed`**: Refined version of `prompt_image_draft` with continuity fixes applied

### Local JSON File
Saved to `phase_2_agents/agent_prompt_review/outputs/review_{episode_id}_{timestamp}.json`:

```json
{
  "episode_id": "E01",
  "title": "Episode Title",
  "reviewed_at": "2025-10-08T...",
  "total_shots": 10,
  "shots_modified": 3,
  "shot_reviews": [
    {
      "shot_id": "S01E01_001",
      "original_prompt": "...",
      "reviewed_prompt": "...",
      "changes_made": [
        "Adjusted lighting to match next shot",
        "Kept flower color consistent"
      ],
      "shot_modified": true,
      "reason_for_modification": "Fixed sunset to sunrise inconsistency",
      "continuity_observations": [
        "Character position consistent",
        "Weather matches across shots"
      ]
    }
  ]
}
```

## Model

Uses **Gemini** (default: `gemini-3.1-pro-preview`) for intelligent continuity review.

## Usage

### Automatic (via API endpoint)
The agent runs automatically when calling `/analyze-shots-mongodb`:

```bash
POST http://localhost:8000/analyze-shots-mongodb?show_id=SHOW123&episode_number=1
```

The response includes:
```json
{
  "analysis_result": {...},
  "mongodb_response": {...},
  "prompt_generation_response": {...},
  "prompt_review_response": {
    "success": true,
    "message": "Successfully reviewed prompts for 10 shots",
    "total_shots": 10,
    "shots_modified": 3
  }
}
```

### Programmatic (Python)
```python
from phase_2_agents.agent_prompt_review import review_image_prompts
from phase_2_agents.agent_shot_strategy import AnnotatedShotList

# After getting annotated_list from Agent 2
updated_list, review_summary = await review_image_prompts(
    annotated_list=annotated_list_with_prompts,
    mongodb_client=mongodb_client,
    scene_description="Overall scene description",
    show_id="SHOW123",
    episode_number=1
)

# Access reviewed prompts
for shot in updated_list.annotated_shots:
    print(f"{shot.shot_id}: {shot.prompt_image_reviewed}")
```

## Configuration

Configure via environment variables:
- `GOOGLE_API_KEY`: Required for Gemini API access
- `MONGODB_ATLAS_URI`: Required for MongoDB updates

## Files

```
phase_2_agents/agent_prompt_review/
├── __init__.py                 # Package initialization
├── prompt_review_agent.py      # Main agent implementation
├── README.md                   # This file
└── outputs/                    # Review JSON files saved here
    └── review_E01_20251008_*.json
```

## Key Features

✅ **Global Continuity Check**: Analyzes all prompts together for overall consistency  
✅ **Pairwise Review**: Checks consecutive shots for micro-continuity  
✅ **Strategy-Aware**: Considers generation strategy (last_frame_seed, multi_shot, generate_new)  
✅ **Minimal Edits**: Only modifies prompts when necessary for continuity  
✅ **Detailed Justification**: Explains each modification clearly  
✅ **MongoDB Integration**: Automatically updates database  
✅ **Local Backup**: Saves detailed review reports as JSON  

## Error Handling

The agent gracefully handles errors:
- If review fails, the original `prompt_image_draft` is preserved
- MongoDB updates are optional (continues if DB unavailable)
- Detailed error logging for debugging
- Falls back to original prompts rather than failing entire pipeline

## Testing

Run the standalone test:
```bash
python phase_2_agents/agent_prompt_review/prompt_review_agent.py
```

Or use the test script:
```bash
python test_prompt_review_agent.py
```

## Author

Created as part of Phase 2 Agent Development - October 2025

