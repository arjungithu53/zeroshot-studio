# Phase 2 Agents

This directory contains AI agents for the second phase of the production project, focusing on advanced shot generation, strategy selection, and continuity management.

## Overview

Phase 2 includes **three AI agents** that work together in a pipeline:

1. **Agent 1 - Shot Strategy Agent**: Analyzes shot lists and determines optimal generation strategies
2. **Agent 2 - Prompt Generation Agent**: Creates cinematic image prompts for each shot
3. **Agent 3 - Prompt Review Agent**: Reviews and refines prompts for visual and narrative continuity

These agents are integrated into the `/analyze-shots-mongodb` API endpoint and execute sequentially.

## Agent Shot Strategy

The `agent_shot_strategy` module provides a LangChain-based AI agent that analyzes episode shot lists and determines optimal generation strategies for each shot.

### Purpose

The Shot Strategy Agent is designed to:

- **Analyze shot continuity**: Detect visual and narrative continuity between consecutive shots
- **Select generation strategies**: Choose between `multi_shot`, `last_frame_seed`, or `generate_new` for each shot
- **Provide reasoning**: Explain why each strategy was chosen with detailed reasoning
- **Maintain quality**: Ensure consistent and logical strategy selection across entire episodes

### Generation Strategies

1. **`generate_new`**: Create completely new content from scratch
   - Used for: First shots, scene changes, new characters/locations
   - When: No previous context or strong discontinuity

2. **`last_frame_seed`**: Use the last frame of the previous shot as a seed
   - Used for: Strong visual continuity, same character/location, smooth transitions
   - When: Previous shot provides good visual foundation

3. **`multi_shot`**: Use a single generated image across multiple shots that share the same environment, characters, and framing
   - Used for: Consecutive shots sharing backgrounds and static character poses, requiring only minor viewpoint or cropping changes
   - When: Minimizing redundant generation and maintaining visual consistency by reusing base images

### Usage Instructions

#### Basic Usage

```python
from phase_2_agents.agent_shot_strategy import ShotStrategyAgent
from phase_2_agents.agent_shot_strategy.data_schema import ShotList, ShotItem
from langchain.llms import OpenAI

# Initialize the agent
llm = OpenAI(temperature=0.1)
agent = ShotStrategyAgent(llm)

# Create a shot list
shot_list = ShotList(
    episode_id="E01",
    title="The Beginning",
    shots=[
        ShotItem(
            shot_id="S01E01_001",
            description="Wide establishing shot of a modern office building",
            duration=4.0,
            scene_number=1,
            sequence_number=1
        ),
        ShotItem(
            shot_id="S01E01_002",
            description="Close-up of protagonist's face",
            duration=3.0,
            scene_number=1,
            sequence_number=2
        )
    ]
)

# Analyze the shot list
annotated_list = agent.analyze_shot_list(shot_list)

# Access results
for shot in annotated_list.annotated_shots:
    print(f"Shot {shot.shot_id}: {shot.generation_strategy}")
    # Get reasoning from versioned structure or legacy field
    reasoning = None
    if shot.image and "v0" in shot.image:
        reasoning = shot.image["v0"].get("reasoning")
    elif hasattr(shot, 'reasoning') and shot.reasoning:
        reasoning = shot.reasoning
    
    if reasoning:
        print(f"Reasoning: {reasoning}")
    print(f"Confidence: {shot.confidence_score}")
```

#### Batch Processing

```python
# Analyze multiple episodes
shot_lists = [shot_list_1, shot_list_2, shot_list_3]
annotated_lists = agent.batch_analyze(shot_lists)

# Get strategy summary
for annotated_list in annotated_lists:
    summary = agent.get_strategy_summary(annotated_list)
    print(f"Episode {summary['episode_id']}: {summary['strategy_distribution']}")
```

#### Validation and Quality Control

```python
# Validate strategy choices
warnings = agent.validate_strategies(annotated_list)
if warnings:
    print("Validation warnings:", warnings)

# Get detailed summary
summary = agent.get_strategy_summary(annotated_list)
print(f"Average confidence: {summary['average_confidence']:.2f}")
print(f"High confidence shots: {summary['high_confidence_shots']}")
```

### Design Choices

#### Continuity Analysis
- **Character continuity**: Detects when the same characters appear in consecutive shots
- **Location continuity**: Identifies when shots share the same setting
- **Action continuity**: Recognizes smooth action transitions
- **Camera continuity**: Detects consistent camera movements and angles

#### Strategy Selection Logic
- **First shots**: Always use `generate_new` (no previous context)
- **Scene changes**: Use `generate_new` (new context required)
- **Strong continuity**: Use `last_frame_seed` (build on previous shot)
- **Shared environments**: Use `multi_shot` (reuse base image with minor viewpoint changes)

#### Quality Assurance
- **Confidence scoring**: Each strategy gets a confidence score (0.0-1.0)
- **Validation**: Built-in validation for consistency and logic
- **Reasoning**: Detailed explanations for each strategy choice
- **Continuity notes**: Analysis of visual and narrative flow

### File Structure

```
agent_shot_strategy/
├── __init__.py              # Package initialization
├── shot_strategy_agent.py   # Main agent implementation
├── data_schema.py           # Pydantic models for input/output
├── prompts.py               # LLM prompt templates and examples
└── utils.py                 # Helper functions and utilities
```

### Dependencies

- **LangChain**: Core agent framework
- **Pydantic**: Data validation and serialization
- **Python 3.8+**: Required for type hints and modern features

### Testing

The agent includes comprehensive testing capabilities:

```python
# Test with example data
from phase_2_agents.agent_shot_strategy.shot_strategy_agent import test_agent_with_example

example_shot_list = test_agent_with_example()
# Use with your LLM instance for testing
```

### Integration

The agent is designed for easy integration with:

- **Phase 1 agents**: Can process output from asset generation agents
- **Video generation pipelines**: Provides strategy guidance for shot creation
- **Quality control systems**: Offers validation and confidence scoring
- **Batch processing**: Handles multiple episodes efficiently

### Future Extensions

The modular design allows for easy extension:

- **Custom continuity detectors**: Add domain-specific continuity analysis
- **Advanced prompting**: Enhance LLM prompts with more examples
- **Strategy optimization**: Learn from successful strategy choices
- **Integration hooks**: Connect with other pipeline components

### Troubleshooting

Common issues and solutions:

1. **Low confidence scores**: Check shot descriptions for clarity and detail
2. **Inconsistent strategies**: Review continuity analysis and prompt examples
3. **JSON parsing errors**: Ensure LLM output follows expected format
4. **Validation warnings**: Review strategy choices for logical consistency

For more detailed information, see the individual module documentation and examples.

---

## Agent Image Prompt Generator (Agent 2)

The `image_prompt_generator_agent` module creates cinematic, detailed image generation prompts for Google Imagen based on shot information and generation strategies.

### Purpose

The Prompt Generation Agent:

- **Generates cinematic prompts**: Creates vivid, filmmaker-quality descriptions for image generation
- **Maintains continuity awareness**: References previous shots and strategies for consistency
- **Incorporates metadata**: Uses shot style, camera movement, and AI notes in prompts
- **Strategy-specific prompting**: Adapts prompt style based on generation strategy
- **Saves to MongoDB**: Updates `image.v0` field with versioned structure for each shot

### Key Features

- **Continuity-Aware**: References seed shots and previous prompts for visual consistency
- **Comprehensive Detail**: Includes lighting, composition, color, atmosphere, and technical specs
- **Professional Language**: Uses cinematography terminology naturally
- **Gemini-Powered**: Leverages Google's Gemini AI for intelligent prompt generation
- **Automatic Integration**: Runs as part of `/analyze-shots-mongodb` pipeline

### Usage

The agent is automatically called in the API flow:

```python
from phase_2_agents.image_prompt_generator_agent import generate_image_prompts_for_shots

# Called after strategy analysis
annotated_list_with_prompts = await generate_image_prompts_for_shots(
    annotated_list=annotated_list,
    mongodb_client=mongodb_client,
    scene_description="Overall scene context",
    show_id=show_id,
    episode_number=episode_number
)

# Access generated prompts
for shot in annotated_list_with_prompts.annotated_shots:
    # Access generated prompts from new versioned structure
    if shot.image and "v0" in shot.image:
        print(f"{shot.shot_id}: {shot.image['v0']['updated_prompt']}")
    elif hasattr(shot, 'prompt_image_draft') and shot.prompt_image_draft:
        print(f"{shot.shot_id}: {shot.prompt_image_draft}")
```

### Output

- **MongoDB Field**: `image.v0` - Versioned structure with cinematic image generation prompt and S3 URLs
- **Local File**: `phase_2_agents/prompts_image/prompts_{episode_id}_{timestamp}.json`

---

## Agent Prompt Review (Agent 3)

The `agent_prompt_review` module reviews and refines image prompts to ensure visual and narrative continuity across shot sequences.

### Purpose

The Prompt Review Agent:

- **Reviews for continuity**: Checks visual consistency across all shots
- **Identifies errors**: Detects lighting, weather, character, and spatial inconsistencies
- **Refines prompts**: Makes minimal edits to fix continuity issues
- **Provides justification**: Explains each modification with detailed reasoning
- **Saves review reports**: Documents all changes and observations

### What It Checks

1. **Visual Consistency**: Lighting, weather, time of day, color palette, atmosphere
2. **Spatial Continuity**: Character blocking, prop positions, screen direction
3. **Directional Consistency**: Sun position, light direction, camera angles
4. **Character Consistency**: Costume, appearance, emotional state
5. **Scene Details**: Background elements, props, established visual details

### Key Features

- **Global Review**: Analyzes all prompts together for overall consistency
- **Pairwise Review**: Checks consecutive shots for micro-continuity
- **Strategy-Aware**: Considers generation strategy in review logic
- **Minimal Edits**: Only modifies when necessary for continuity
- **Detailed Reports**: Saves comprehensive review data as JSON

### Usage

The agent is automatically called after prompt generation:

```python
from phase_2_agents.agent_prompt_review import review_image_prompts

# Called after prompt generation
updated_list, review_summary = await review_image_prompts(
    annotated_list=annotated_list_with_prompts,
    mongodb_client=mongodb_client,
    scene_description="Overall scene context",
    show_id=show_id,
    episode_number=episode_number
)

# Access reviewed prompts
for shot in updated_list.annotated_shots:
    # Access reviewed prompts from new versioned structure
    if shot.image and "v1" in shot.image:
        print(f"{shot.shot_id}: {shot.image['v1']['updated_prompt']}")
    elif hasattr(shot, 'prompt_image_reviewed') and shot.prompt_image_reviewed:
        print(f"{shot.shot_id}: {shot.prompt_image_reviewed}")
```

### Output

- **MongoDB Field**: `image.v1` - Versioned structure with refined prompt, continuity fixes, and S3 URLs
- **Local File**: `phase_2_agents/agent_prompt_review/outputs/review_{episode_id}_{timestamp}.json`

---

## Pipeline Flow

The complete Phase 2 pipeline executes as follows:

```
POST /analyze-shots-mongodb
  │
  ├─► Phase 1: Strategy Agent (Agent 1)
  │   ├─ Analyzes shot list for continuity
  │   ├─ Determines generation strategy per shot
  │   ├─ Saves to MongoDB: generation_strategy, reasoning, confidence_score
  │   └─ Output: annotated_list with strategies
  │
  ├─► Phase 2: Prompt Generation Agent (Agent 2)
  │   ├─ Reads strategies from Phase 1
  │   ├─ Generates cinematic image prompts
  │   ├─ Saves to MongoDB: image.v0 (versioned structure)
  │   └─ Output: annotated_list with draft prompts
  │
  └─► Phase 3: Prompt Review Agent (Agent 3)
      ├─ Reads draft prompts from Phase 2
      ├─ Reviews for visual and narrative continuity
      ├─ Refines prompts with minimal edits
      ├─ Saves to MongoDB: image.v1 (versioned structure)
      └─ Output: annotated_list with reviewed prompts + review report
```

## Testing

Each agent can be tested individually:

```bash
# Test Agent 1 - Strategy Agent
python phase_2_agents/agent_shot_strategy/example_usage.py

# Test Agent 2 - Prompt Generator
python test_image_prompt_agent.py

# Test Agent 3 - Prompt Review
python test_prompt_review_agent.py
```

Or test the complete pipeline via API:

```bash
# Start the server
python start_server.py

# Call the integrated endpoint
curl -X POST "http://localhost:8000/analyze-shots-mongodb?show_id=SHOW123&episode_number=1" \
  -H "Content-Type: application/json" \
  -d @postman_mongodb_request_body.json
```

## Directory Structure

```
phase_2_agents/
├── agent_shot_strategy/           # Agent 1: Strategy Agent
│   ├── __init__.py
│   ├── shot_strategy_agent.py
│   ├── data_schema.py
│   ├── prompts.py
│   ├── utils.py
│   └── mongodb_utils.py
├── image_prompt_generator_agent.py # Agent 2: Prompt Generator
├── agent_prompt_review/           # Agent 3: Prompt Review
│   ├── __init__.py
│   ├── prompt_review_agent.py
│   ├── README.md
│   └── outputs/
├── prompts_image/                 # Output: Generated prompts
└── README.md                      # This file
```

## MongoDB Schema

Phase 2 agents add the following fields to the shots collection:

```javascript
{
  "shot_id": "S01E01_001",
  "show_id": "SHOW123",
  "episode_number": 1,
  "description": "Wide shot of lakeside cottage",
  
  // Added by Agent 1
  "generation_strategy": "generate_new",
  "reasoning": "First shot establishes scene",
  "confidence_score": 0.95,
  "seed_shot_id": null,
  
  // Added by Agent 2 - NEW VERSIONED STRUCTURE
  "image": {
    "v0": {
      "updated_prompt": "Wide cinematic shot of rustic lakeside cottage at golden hour...",
      "changes_made": "Initial image prompt generated by Agent 2",
      "reasoning": "AI-generated prompt based on shot description and strategy",
      "generated_images_s3": []
    }
  },
  
  // Added by Agent 3 - NEW VERSIONED STRUCTURE
  "image": {
    "v0": { /* ... existing v0 data ... */ },
    "v1": {
      "updated_prompt": "Wide cinematic shot of rustic lakeside cottage at golden hour sunset...",
      "changes_made": "Prompt reviewed and refined by Agent 3 for continuity",
      "reasoning": "Review agent applied continuity fixes and improvements",
      "generated_images_s3": []
    }
  }
}
```

## Environment Variables

Required for all Phase 2 agents:

```bash
# .env file
GOOGLE_API_KEY=your_gemini_api_key_here
MONGODB_ATLAS_URI=mongodb+srv://...
MONGODB_DATABASE_NAME=production
MONGODB_SHOTS_COLLECTION=shots
```

## Error Handling

All agents include robust error handling:

- **Graceful Degradation**: Pipeline continues even if individual agents fail
- **Detailed Logging**: Comprehensive logs for debugging
- **MongoDB Resilience**: Continues without DB if unavailable
- **Fallback Behavior**: Uses previous data if generation fails

## Future Extensions

Potential improvements:

- **Character Library**: Maintain consistent character descriptions across episodes
- **Location Database**: Track and reuse location descriptions
- **Style Consistency**: Learn and apply show-specific visual styles
- **Interactive Review**: Allow manual refinement of reviewed prompts
- **A/B Testing**: Compare different prompt generation strategies

---

For detailed documentation on individual agents, see their respective README files and module documentation.
