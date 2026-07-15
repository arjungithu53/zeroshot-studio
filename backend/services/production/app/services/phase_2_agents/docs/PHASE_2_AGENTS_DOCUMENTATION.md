# Phase 2 Agents Documentation

## Overview

Phase 2 Agents are AI-powered systems that analyze shot lists, generate cinematic image prompts, and review them for visual and narrative continuity. These agents work together in a sequential pipeline to process episode shot data and prepare it for image generation.

## Architecture

```
Phase 2 Pipeline:
1. Shot Strategy Agent (Agent 1) → Analyzes shots and determines generation strategies
2. Image Prompt Generator Agent (Agent 2) → Creates cinematic image prompts
3. Prompt Review Agent (Agent 3) → Reviews and refines prompts for continuity
```

## API Endpoints

### Base URL
```
http://localhost:8000
```

### 1. Health Check
- **Endpoint**: `GET /health`
- **Description**: Check server status and configuration
- **Response**:
```json
{
  "status": "healthy",
  "agent": "ready",
  "mongodb": "connected",
  "google_api_configured": true,
  "mongodb_configured": true
}
```

### 2. Analyze Shots (Basic)
- **Endpoint**: `POST /analyze-shots`
- **Description**: Analyze shot list and return generation strategies (no MongoDB save)
- **Request Body**:
```json
{
  "episode_id": "E01",
  "title": "The Beginning",
  "scene_description": "Optional overall scene context",
  "shots": [
    {
      "shot_id": "S01E01_001",
      "description": "Wide establishing shot of a modern office building",
      "duration": 4.0,
      "scene_number": 1,
      "sequence_number": 1,
      "shot_style": "wide_shot",
      "camera_movement": "static",
      "source_type": "generated",
      "optimized_ai_notes": "Golden hour lighting"
    }
  ]
}
```
- **Response**:
```json
{
  "episode_id": "E01",
  "title": "The Beginning",
  "annotated_shots": [
    {
      "shot_id": "S01E01_001",
      "description": "Wide establishing shot of a modern office building",
      "generation_strategy": "generate_new",
      "reasoning": "First shot establishes scene",
      "confidence_score": 0.95,
      "continuity_notes": "Opening shot"
    }
  ],
  "strategy_summary": {
    "generate_new": 1,
    "last_frame_seed": 0,
    "multi_shot": 0
  }
}
```

### 3. Analyze Shots with MongoDB (Full Pipeline)
- **Endpoint**: `POST /analyze-shots-mongodb?show_id={show_id}&episode_number={episode_number}`
- **Description**: Complete Phase 2 pipeline with MongoDB integration
- **Query Parameters**:
  - `show_id` (string): Show identifier
  - `episode_number` (integer): Episode number
- **Request Body**: Same as `/analyze-shots`
- **Response**:
```json
{
  "analysis_result": {
    "episode_id": "E01",
    "annotated_shots": [...],
    "strategy_summary": {...}
  },
  "mongodb_response": {
    "success": true,
    "message": "Successfully saved 25 shots to MongoDB Atlas",
    "inserted_ids": ["..."]
  },
  "prompt_generation_response": {
    "success": true,
    "message": "Successfully generated image prompts for 25 shots",
    "prompts_generated": 25
  },
  "prompt_review_response": {
    "success": true,
    "message": "Successfully reviewed and refined prompts",
    "prompts_reviewed": 25
  }
}
```

### 4. MongoDB Operations

#### Get Collection Statistics
- **Endpoint**: `GET /mongodb/stats`
- **Description**: Get MongoDB collection statistics
- **Response**:
```json
{
  "success": true,
  "stats": {
    "total_shots": 150,
    "strategy_distribution": {
      "generate_new": 45,
      "last_frame_seed": 30,
      "multi_shot": 25
    },
    "source_distribution": {
      "generated": 120,
      "uploaded": 30
    }
  }
}
```

#### Get Shots by Episode
- **Endpoint**: `GET /mongodb/shots/{show_id}/{episode_number}`
- **Description**: Retrieve shots for a specific episode
- **Response**:
```json
{
  "success": true,
  "shots": [...],
  "count": 25
}
```

## MongoDB Schema

### Shots Collection Schema

```javascript
{
  "_id": ObjectId("..."),
  "shot_id": "S01E01_001",                    // Unique shot identifier
  "show_id": "SHOW123",                      // Show identifier
  "episode_number": 1,                        // Episode number
  "scene_number": 1,                         // Scene number
  "shot_number": 1,                          // Shot number within scene
  "description": "Wide establishing shot...", // Shot description
  "shot_style": "wide_shot",                 // Shot style (optional)
  "camera_movement": "static",               // Camera movement (optional)
  "duration": 4.0,                          // Shot duration in seconds
  "source_type": "generated",               // "generated" or "uploaded"
  "uploaded_image_id": ObjectId("..."),     // If source_type is "uploaded"
  "generated_image_id": ObjectId("..."),    // If source_type is "generated"
  "generated_video_id": ObjectId("..."),    // Generated video reference
  "optimized_ai_notes": "Golden hour...",    // AI optimization notes
  
  // Phase 2 Agent 1 Outputs (Shot Strategy Agent)
  "generation_strategy": "generate_new",     // "generate_new", "last_frame_seed", "multi_shot"
  "reasoning": "First shot establishes...", // Strategy reasoning
  "confidence_score": 0.95,                 // Confidence score (0.0-1.0)
  "continuity_notes": "Opening shot",        // Continuity analysis
  "seed_shot_id": "S01E01_002",             // For last_frame_seed strategy
  "seed_frame_shot_id": "S01E01_002",       // Specific frame reference
  
  // Phase 2 Agent 2 Outputs (Image Prompt Generator) - NEW VERSIONED STRUCTURE
  "image": {
    "v0": {
      "updated_prompt": "Cinematic wide shot...", // AI-generated image prompt
      "changes_made": "Initial image prompt generated by Agent 2",
      "reasoning": "AI-generated prompt based on shot description and strategy",
      "generated_images_s3": []
    }
  },
  
  // Phase 2 Agent 3 Outputs (Prompt Review Agent) - NEW VERSIONED STRUCTURE
  "image": {
    "v0": { /* ... existing v0 data ... */ },
    "v1": {
      "updated_prompt": "Cinematic wide shot...", // Reviewed and refined prompt
      "changes_made": "Prompt reviewed and refined by Agent 3 for continuity",
      "reasoning": "Review agent applied continuity fixes and improvements",
      "generated_images_s3": []
    }
  }
  
  // Phase 3 Inputs (Manual) - NOW PART OF VERSIONED STRUCTURE
  // S3 URLs are now stored in image.v0.generated_images_s3 and image.v1.generated_images_s3
  
  // Phase 3 Outputs (Video Generation Agent) - NEW VERSIONED STRUCTURE
  "video": {
    "v0": {
      "updated_prompt": "Cinematic video...", // AI-generated video prompt
      "changes_made": "Initial video prompt generated by Agent",
      "reasoning": "AI-generated video prompt based on image prompts and continuity",
      "generated_videos_s3": []
    }
  }
  
  // Phase 3 Review Outputs (Video Prompt Review Agent)
  "video_prompt_reviewed_A": {              // Reviewed video prompt
    "draft_prompt": "...",
    "updated_prompt": "...",
    "changes_made": "...",
    "reasoning": "...",
    "timestamp": "2025-01-01T12:00:00"
  }
}
```

### Indexes

The system automatically creates the following indexes for optimal performance:

```javascript
// Compound index for episode queries
db.shots.createIndex({
  "show_id": 1,
  "episode_number": 1,
  "scene_number": 1,
  "shot_number": 1
})

// Single field indexes
db.shots.createIndex({"generation_strategy": 1})
db.shots.createIndex({"source_type": 1})
db.shots.createIndex({"seed_shot_id": 1})
```

## Agent Details

### Agent 1: Shot Strategy Agent

**Purpose**: Analyzes shot lists and determines optimal generation strategies for each shot.

**Key Features**:
- Continuity analysis between consecutive shots
- Strategy selection: `generate_new`, `last_frame_seed`, `multi_shot`
- Confidence scoring and reasoning
- Visual and narrative continuity detection

**Generation Strategies**:

1. **`generate_new`**: Create completely new content
   - Used for: First shots, scene changes, new characters/locations
   - When: No previous context or strong discontinuity

2. **`last_frame_seed`**: Use previous shot's last frame as seed
   - Used for: Strong visual continuity, same character/location
   - When: Previous shot provides good visual foundation

3. **`multi_shot`**: Reuse single image across multiple shots
   - Used for: Consecutive shots sharing environment/characters
   - When: Minimizing redundant generation

**Input**: Shot list with descriptions, durations, scene numbers
**Output**: Annotated shot list with strategies, reasoning, confidence scores

### Agent 2: Image Prompt Generator Agent

**Purpose**: Creates cinematic, detailed image generation prompts using Gemini AI.

**Key Features**:
- Generates vivid, filmmaker-quality descriptions
- Maintains continuity awareness with previous shots
- Incorporates shot metadata (style, camera movement, AI notes)
- Strategy-specific prompting
- Saves to MongoDB `prompt_image_draft` field

**Input**: Annotated shot list from Agent 1
**Output**: Enhanced shot list with versioned `image.v0` field containing:
- `updated_prompt`: The generated image prompt
- `changes_made`: Description of what was generated
- `reasoning`: Strategy reasoning (moved from top-level field)
- `generated_images_s3`: Array for future S3 URLs

**Example Output**:
```
"Cinematic wide establishing shot of a modern glass office building at golden hour, 
warm sunlight reflecting off the windows, professional architectural photography, 
clean lines and geometric shapes, urban environment"
```

### Agent 3: Prompt Review Agent

**Purpose**: Reviews and refines image prompts for visual and narrative continuity.

**Key Features**:
- Global review of all prompts together
- Pairwise review of consecutive shots
- Visual consistency checking (lighting, weather, time of day)
- Spatial continuity (character blocking, prop positions)
- Character consistency (costume, appearance, emotional state)
- Minimal edits with detailed justification

**What It Checks**:
1. Visual Consistency: Lighting, weather, color palette, atmosphere
2. Spatial Continuity: Character blocking, prop positions, screen direction
3. Directional Consistency: Sun position, light direction, camera angles
4. Character Consistency: Costume, appearance, emotional state
5. Scene Details: Background elements, props, established visual details

**Input**: Shot list with draft prompts from Agent 2
**Output**: Enhanced shot list with versioned `image.v1` field containing:
- `updated_prompt`: The reviewed and refined image prompt
- `changes_made`: List of continuity fixes applied
- `reasoning`: Review agent reasoning for changes (enhanced from review results)
- `generated_images_s3`: Array for future S3 URLs

## Data Models

### Request Models

#### ShotItemRequest
```python
class ShotItemRequest(BaseModel):
    shot_id: str                              # Unique shot identifier
    description: str                          # Shot description
    duration: Optional[float]                  # Duration in seconds
    scene_number: Optional[int]                # Scene number
    sequence_number: Optional[int]             # Sequence number
    shot_style: Optional[str]                 # Shot style
    camera_movement: Optional[str]            # Camera movement
    source_type: str = "generated"            # "generated" or "uploaded"
    uploaded_image_id: Optional[str]          # ObjectId if uploaded
    generated_image_id: Optional[str]          # ObjectId if generated
    generated_video_id: Optional[str]         # ObjectId of video
    optimized_ai_notes: Optional[str]         # AI optimization notes
```

#### ShotListRequest
```python
class ShotListRequest(BaseModel):
    episode_id: str                           # Episode identifier
    title: Optional[str]                      # Episode title
    shots: List[ShotItemRequest]              # List of shots
    scene_description: Optional[str]          # Overall scene context
```

### Response Models

#### ShotStrategyResponse
```python
class ShotStrategyResponse(BaseModel):
    episode_id: str
    title: Optional[str]
    annotated_shots: List[Dict[str, Any]]     # Annotated shots with strategies
    overall_continuity_notes: Optional[str]   # High-level continuity notes
    strategy_summary: Dict[str, int]         # Strategy distribution
    analysis_summary: Dict[str, Any]         # Analysis summary
```

## Environment Variables

Required environment variables in `.env`:

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here
MONGODB_ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/

# Optional (with defaults)
MONGODB_DATABASE_NAME=production
MONGODB_SHOTS_COLLECTION=shots
```

## Error Handling

All agents include robust error handling:

- **Graceful Degradation**: Pipeline continues even if individual agents fail
- **Detailed Logging**: Comprehensive logs for debugging
- **MongoDB Resilience**: Continues without DB if unavailable
- **Fallback Behavior**: Uses previous data if generation fails

## Testing

### Individual Agent Testing
```bash
# Test Agent 1 - Strategy Agent
python phase_2_agents/agent_shot_strategy/example_usage.py

# Test Agent 2 - Prompt Generator
python test_image_prompt_agent.py

# Test Agent 3 - Prompt Review
python test_prompt_review_agent.py
```

### Full Pipeline Testing
```bash
# Start the server
python start_server.py

# Test complete pipeline
curl -X POST "http://localhost:8000/analyze-shots-mongodb?show_id=SHOW123&episode_number=1" \
  -H "Content-Type: application/json" \
  -d @postman_mongodb_request_body.json
```

## Output Files

### Local JSON Files
- **Location**: `phase_2_agents/prompts_image/`
- **Format**: `prompts_{episode_id}_{timestamp}.json`
- **Content**: Generated image prompts with metadata

### Review Output Files
- **Location**: `phase_2_agents/agent_prompt_review/outputs/`
- **Format**: `review_{episode_id}_{timestamp}.json`
- **Content**: Review reports with changes and reasoning

## Integration Flow

```
1. POST /analyze-shots-mongodb
   ↓
2. Agent 1: Shot Strategy Analysis
   ├─ Analyzes shot continuity
   ├─ Determines generation strategies
   ├─ Saves to MongoDB: generation_strategy, reasoning, confidence_score
   └─ Output: annotated_list with strategies
   ↓
3. Agent 2: Image Prompt Generation
   ├─ Reads strategies from Agent 1
   ├─ Generates cinematic image prompts
   ├─ Saves to MongoDB: image.v0 (versioned structure)
   └─ Output: annotated_list with versioned image prompts
   ↓
4. Agent 3: Prompt Review
   ├─ Reads draft prompts from Agent 2
   ├─ Reviews for visual and narrative continuity
   ├─ Refines prompts with minimal edits
   ├─ Saves to MongoDB: image.v1 (versioned structure)
   └─ Output: annotated_list with reviewed prompts + review report
```

## Best Practices

1. **Run Complete Pipeline**: Use `/analyze-shots-mongodb` for full Phase 2 processing
2. **Provide Scene Context**: Include `scene_description` for better prompts
3. **Monitor Logs**: Check logs for any errors or warnings
4. **Review Outputs**: Inspect generated prompts before using them
5. **Database Indexes**: Ensure MongoDB indexes are created for performance

## Troubleshooting

### Common Issues

1. **Low confidence scores**
   - **Cause**: Unclear shot descriptions
   - **Solution**: Provide more detailed shot descriptions

2. **Inconsistent strategies**
   - **Cause**: Poor continuity analysis
   - **Solution**: Review shot sequence and descriptions

3. **JSON parsing errors**
   - **Cause**: LLM output format issues
   - **Solution**: Check Gemini API key and model availability

4. **MongoDB connection failed**
   - **Cause**: Invalid connection string
   - **Solution**: Verify `MONGODB_ATLAS_URI` in `.env`

5. **Missing prompts**
   - **Cause**: Agent 2 or 3 failed
   - **Solution**: Check logs and retry individual agents

## Future Enhancements

- **Character Library**: Maintain consistent character descriptions
- **Location Database**: Track and reuse location descriptions
- **Style Consistency**: Learn and apply show-specific visual styles
- **Interactive Review**: Allow manual refinement of reviewed prompts
- **A/B Testing**: Compare different prompt generation strategies

## Related Documentation

- [Phase 3 Agents Documentation](PHASE_3_AGENTS_DOCUMENTATION.md)
- [MongoDB Atlas Setup](documents/MONGODB_ATLAS_SETUP.md)
- [API Setup Guide](API_SETUP_GUIDE.md)
- [Quick Start Guide](documents/QUICK_START_GUIDE.md)
