# Phase 3 AI Agent: Video Generation Agent

## Overview

The **Video Generation Agent** is Phase 3 of the production project's AI pipeline. This agent fetches data from MongoDB, reads all previous Phase 2 agent outputs, and generates structured prompts for video generation for each shot.

## Architecture

```
Phase 3 Flow:
1. Fetch shot data from MongoDB (including all Phase 2 outputs)
2. For each shot:
   - Read: scene_description, optimized_ai_notes, generation_strategy, 
     reasoning, confidence_score, prompt_image_draft, prompt_image_reviewed, 
     generated_images_s3
   - Generate comprehensive video prompt using Gemini
   - Consider continuity from previous shots
3. Save to MongoDB (field: prompt_video_draft)
4. Save local copy to phase_3_agents/output/
```

## Key Features

✅ **Comprehensive Data Fetching**: Retrieves all Phase 2 outputs from MongoDB  
✅ **Intelligent Video Prompts**: Uses Gemini to generate cinematic video prompts  
✅ **Continuity Awareness**: Considers previous shot prompts for visual continuity  
✅ **Strategy Integration**: Respects generation strategies from Phase 2  
✅ **Dual Storage**: Saves to both MongoDB and local files  
✅ **S3 Reference Support**: Incorporates manually uploaded S3 image URLs

## New Database Fields

### `generated_images_s3` (List[String])
- **Type**: Array of strings
- **Description**: List of manually uploaded S3 URLs for generated images
- **Usage**: Populated manually by the user after images are uploaded to S3
- **Example**: `["https://s3.amazonaws.com/bucket/shot1.png", "https://s3.amazonaws.com/bucket/shot2.png"]`

### `prompt_video_draft` (String)
- **Type**: String
- **Description**: AI-generated cinematic video generation prompt
- **Usage**: Automatically populated by the Video Generation Agent
- **Example**: "A cinematic establishing shot of a futuristic city at sunset, smooth camera push-in movement..."

## Installation

No additional dependencies required beyond the existing project setup.

## Usage

### 1. Direct Agent Usage

```python
import asyncio
from phase_3_agents.agent_video_generation import VideoGenerationAgent
from phase_2_agents.agent_shot_strategy import MongoDBAtlasClient

async def generate_video_prompts():
    # Initialize MongoDB client
    mongodb_client = MongoDBAtlasClient(
        database_name="production",
        shots_collection="shots"
    )
    
    # Initialize Video Generation Agent
    agent = VideoGenerationAgent(
        model_name="gemini-3.1-pro-preview",
        temperature=0.7,
        enable_saving=True
    )
    
    # Generate video prompts for an episode
    result = await agent.generate_video_prompts_for_episode(
        show_id="E01",
        episode_number=1,
        mongodb_client=mongodb_client,
        scene_description="Optional overall scene context"
    )
    
    print(f"Generated {result['video_prompts_saved']} video prompts")

asyncio.run(generate_video_prompts())
```

### 2. API Endpoint Usage

**Endpoint:** `POST /image-video-generation`

**Request Body:**
```json
{
  "show_id": "E01",
  "episode_number": 1,
  "scene_description": "Optional scene context for better prompts"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully generated video prompts for 25 shots",
  "video_prompts_saved": 25,
  "total_shots": 25,
  "local_file": "phase_3_agents/output/E01_1_video_prompts_20251008_135744.json",
  "data_preview": [
    {
      "shot_id": "S01E01_001",
      "description": "Wide establishing shot of city",
      "generation_strategy": "generate_new",
      "video_prompt": "A cinematic establishing shot...",
      "estimated_duration_seconds": 4,
      "reference_images_s3": ["https://s3.amazonaws.com/..."]
    }
  ]
}
```

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/image-video-generation" \
  -H "Content-Type: application/json" \
  -d '{
    "show_id": "E01",
    "episode_number": 1,
    "scene_description": "A dramatic opening scene in a modern city"
  }'
```

**Example with Python requests:**
```python
import requests

response = requests.post(
    "http://localhost:8000/image-video-generation",
    json={
        "show_id": "E01",
        "episode_number": 1,
        "scene_description": "A dramatic opening scene"
    }
)

result = response.json()
print(f"Status: {result['status']}")
print(f"Prompts saved: {result['video_prompts_saved']}")
```

## Video Prompt Generation Process

The agent follows this process for each shot:

1. **Data Collection**
   - Fetches shot data from MongoDB
   - Reads all Phase 2 outputs
   - Gathers reference image URLs (if available)

2. **Prompt Construction**
   ```
   Input to Gemini:
   - Scene description (overall context)
   - Shot information (ID, description, style, camera movement, duration)
   - Optimized AI notes
   - Generation strategy + reasoning + confidence score
   - Image prompts (reviewed > draft)
   - Reference images from S3
   - Previous shot's video prompt (for continuity)
   ```

3. **Gemini Processing**
   - Generates comprehensive video prompt
   - Suggests estimated duration
   - Ensures continuity with previous shots
   - Respects generation strategy

4. **Output Structure**
   ```json
   {
     "shot_id": "S01E01_001",
     "video_prompt": "Detailed cinematic video description...",
     "estimated_duration_seconds": 4
   }
   ```

5. **Storage**
   - Save to MongoDB: `prompt_video_draft` field
   - Save locally: `phase_3_agents/output/{show_id}_{episode}_video_prompts_{timestamp}.json`

## Gemini Prompt Structure

The agent uses a carefully crafted prompt structure:

```
You are a cinematic video generation assistant.
Generate a concise but visually rich video generation prompt for this shot.

=== SCENE DESCRIPTION ===
[Overall context]

=== SHOT INFORMATION ===
Shot ID: [shot_id]
Description: [description]
Shot Style: [shot_style]
Camera Movement: [camera_movement]
Duration: [duration]

=== OPTIMIZED AI NOTES ===
[optimized_ai_notes]

=== SHOT GENERATION STRATEGY ===
Strategy: [generation_strategy]
Reasoning: [reasoning]
Confidence Score: [confidence_score]

=== IMAGE PROMPT ===
[prompt_image_reviewed or prompt_image_draft]

=== REFERENCE IMAGES ===
[S3 URLs]

=== PREVIOUS SHOT VIDEO PROMPT ===
[For continuity]

=== YOUR TASK ===
Generate a comprehensive video generation prompt that:
1. Ensures visual continuation from previous frames
2. Maintains the same tone and narrative flow
3. Incorporates camera movement and shot style
4. Considers the generation strategy
5. Suggests suitable video length
6. Uses professional cinematography terminology
7. Is detailed enough for video generation models

Return JSON:
{
  "shot_id": "...",
  "video_prompt": "...",
  "estimated_duration_seconds": <int>
}
```

## Testing

Run the comprehensive test suite:

```bash
# Test direct agent and pipeline
python test_video_generation_agent.py

# Test API endpoint (requires server running)
python api_server.py  # In one terminal
API_TEST=1 python test_video_generation_agent.py  # In another terminal
```

## Output Files

### Local JSON Files
Location: `phase_3_agents/output/`

Format: `{show_id}_{episode_number}_video_prompts_{timestamp}.json`

Example structure:
```json
{
  "show_id": "E01",
  "episode_number": 1,
  "episode_id": "E01",
  "generated_at": "2025-10-08T13:57:44.123456",
  "total_shots": 25,
  "video_prompts": [
    {
      "shot_id": "S01E01_001",
      "description": "Wide establishing shot of city",
      "generation_strategy": "generate_new",
      "video_prompt": "A cinematic establishing shot of a sprawling modern metropolis at golden hour...",
      "estimated_duration_seconds": 4,
      "reference_images_s3": [
        "https://s3.amazonaws.com/bucket/shot1.png"
      ]
    }
  ]
}
```

## Configuration

Environment variables (in `.env`):

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key
MONGODB_ATLAS_URI=your_mongodb_connection_string

# Optional
MONGODB_DATABASE_NAME=production  # default
MONGODB_SHOTS_COLLECTION=shots  # default
```

## Agent Configuration

```python
VideoGenerationAgent(
    model_name="gemini-3.1-pro-preview",  # Gemini model
    temperature=0.7,                     # Creativity level (0.0-1.0)
    max_tokens=4096,                     # Max output tokens
    enable_saving=True,                  # Save to local files
    output_dir="phase_3_agents/output"  # Output directory
)
```

## Integration with Pipeline

Phase 3 integrates seamlessly with the existing pipeline:

```
Phase 1 (Legacy) → Phase 2 → Phase 3
                     ↓          ↓
                  MongoDB   MongoDB
                  (image    (video
                  prompts)  prompts)
```

**Complete Flow:**
1. Phase 2 Agent 1: Shot Strategy Analysis → `generation_strategy`, `reasoning`, `confidence_score`
2. Phase 2 Agent 2: Image Prompt Generation → `prompt_image_draft`
3. Phase 2 Agent 3: Image Prompt Review → `prompt_image_reviewed`
4. **Manual Step**: User uploads generated images to S3 → `generated_images_s3`
5. **Phase 3**: Video Prompt Generation → `prompt_video_draft`

## Error Handling

The agent includes robust error handling:

- **MongoDB Connection Errors**: Graceful failure with detailed error messages
- **Missing Data**: Uses fallback prompts if fields are missing
- **Gemini API Errors**: Fallback to description-based prompts
- **JSON Parsing Errors**: Extracts JSON or uses raw response
- **Network Issues**: Retries and detailed logging

## Logging

The agent provides detailed logging at each step:

```
INFO: Fetching shot data for show E01, episode 1
INFO: Successfully fetched 25 shots from MongoDB
INFO: Generating video prompt for shot: S01E01_001
INFO: Successfully generated video prompt for S01E01_001
INFO: ✅ Saved video prompt for shot S01E01_001 to MongoDB
INFO: Video prompts saved to: phase_3_agents/output/E01_1_video_prompts_20251008_135744.json
INFO: ✅ Video prompt generation completed: 25/25 saved to MongoDB
```

## API Integration

The Phase 3 endpoint is automatically added to the main API server:

**Available at:** `http://localhost:8000/image-video-generation`

**Documentation:** `http://localhost:8000/docs` (FastAPI auto-generated)

**Health Check:** Verify API status at `http://localhost:8000/health`

## Best Practices

1. **Run Phase 2 First**: Ensure Phase 2 has completed and populated MongoDB
2. **Upload Images to S3**: Manually populate `generated_images_s3` before Phase 3
3. **Provide Scene Context**: Include `scene_description` for better prompts
4. **Monitor Logs**: Check logs for any errors or warnings
5. **Review Outputs**: Inspect generated video prompts before using them

## Troubleshooting

### No shots found
- **Cause**: Phase 2 hasn't been run or MongoDB is empty
- **Solution**: Run Phase 2 first to populate MongoDB

### Empty video prompts
- **Cause**: Gemini API key missing or invalid
- **Solution**: Check `GOOGLE_API_KEY` in `.env`

### MongoDB connection failed
- **Cause**: Invalid connection string or network issue
- **Solution**: Verify `MONGODB_ATLAS_URI` in `.env`

### Missing reference images
- **Cause**: `generated_images_s3` field not populated
- **Solution**: This is normal - field is optional and manually populated

## Future Enhancements

- [ ] Batch processing with parallel Gemini calls
- [ ] Advanced continuity tracking across sequences
- [ ] Integration with actual video generation APIs
- [ ] Support for custom prompt templates
- [ ] Webhook notifications on completion
- [ ] Video preview generation

## Related Documentation

- [Phase 2 Agents Documentation](../phase_2_agents/README.md)
- [MongoDB Atlas Setup](../documents/MONGODB_ATLAS_SETUP.md)
- [API Setup Guide](../API_SETUP_GUIDE.md)
- [Quick Start Guide](../documents/QUICK_START_GUIDE.md)

## License

Part of the production project.

## Support

For issues or questions, contact the development team.

