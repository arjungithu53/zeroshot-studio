# Phase 3 Video Generation Agent - Quick Start Guide

## Prerequisites

✅ Phase 2 completed and MongoDB populated with shot data  
✅ `GOOGLE_API_KEY` configured in `.env`  
✅ `MONGODB_ATLAS_URI` configured in `.env`  
✅ (Optional) Images uploaded to S3 - URLs can be provided in the request

## Quick Start (5 minutes)

### Option 1: Using the API (Recommended)

1. **Start the API server**
   ```bash
   python api_server.py
   ```

2. **Send a POST request**
   
   **Basic (without S3 URLs):**
   ```bash
   curl -X POST "http://localhost:8000/image-video-generation" \
     -H "Content-Type: application/json" \
     -d '{
       "show_id": "E01",
       "episode_number": 1,
       "scene_description": "A dramatic opening scene in a modern city"
     }'
   ```
   
   **With S3 URLs (Recommended):**
   ```bash
   curl -X POST "http://localhost:8000/image-video-generation" \
     -H "Content-Type: application/json" \
     -d '{
       "show_id": "E01",
       "episode_number": 1,
       "scene_description": "A dramatic opening scene in a modern city",
       "shots_with_s3_images": [
         {
           "shot_id": "S01E01_001",
           "s3_urls": ["https://s3.amazonaws.com/your-bucket/shot1.png"]
         },
         {
           "shot_id": "S01E01_002",
           "s3_urls": ["https://s3.amazonaws.com/your-bucket/shot2.png"]
         }
       ]
     }'
   ```

3. **Check the response**
   ```json
   {
     "status": "success",
     "message": "Successfully generated video prompts for 25 shots",
     "video_prompts_saved": 25,
     "total_shots": 25,
     "s3_urls_updated": 2,
     "local_file": "phase_3_agents/output/E01_1_video_prompts_20251008_135744.json"
   }
   ```
   
   Note: `s3_urls_updated` shows how many shots had S3 URLs saved to MongoDB

4. **View results**
   - **MongoDB**: Check the `prompt_video_draft` field for each shot
   - **Local file**: Check `phase_3_agents/output/` for the JSON file

### Option 2: Using Python Directly

```python
import asyncio
from phase_3_agents.agent_video_generation import generate_video_prompts_pipeline
from phase_2_agents.agent_shot_strategy import MongoDBAtlasClient

async def main():
    # Connect to MongoDB
    mongodb_client = MongoDBAtlasClient()
    
    # Generate video prompts
    result = await generate_video_prompts_pipeline(
        show_id="E01",
        episode_number=1,
        mongodb_client=mongodb_client,
        scene_description="Optional scene context"
    )
    
    print(f"✅ Generated {result['video_prompts_saved']} video prompts")
    print(f"📁 Saved to: {result['local_file']}")

asyncio.run(main())
```

## Testing Your Setup

Run the test suite to verify everything works:

```bash
python test_video_generation_agent.py
```

Expected output:
```
✅ Connected to MongoDB
✅ Agent initialized
✅ Found 25 shots
✅ Video prompt generated
✅ Successfully saved video prompt to MongoDB
✅ Direct Agent Test Completed Successfully!
```

## What Gets Generated?

For each shot, the agent generates a comprehensive video prompt like:

```
A cinematic establishing shot of a sprawling modern metropolis at golden hour. 
The camera executes a smooth push-in movement, transitioning from a wide aerial 
perspective to a mid-range view of the city skyline. Dramatic volumetric lighting 
creates god rays through atmospheric haze. Color palette: warm oranges and cool 
blues with high contrast. Shot on 35mm film with shallow depth of field, creating 
beautiful bokeh in the background. Professional cinematography with precise framing 
following the rule of thirds.
```

## Where Are Results Saved?

### 1. MongoDB
- **Collection**: `shots`
- **Field**: `prompt_video_draft`
- **Query**: Find by `show_id` and `episode_number`

Example MongoDB query:
```javascript
db.shots.find({
  "show_id": "E01",
  "episode_number": 1
})
```

### 2. Local Files
- **Directory**: `phase_3_agents/output/`
- **Format**: `{show_id}_{episode_number}_video_prompts_{timestamp}.json`
- **Example**: `E01_1_video_prompts_20251008_135744.json`

## Common Issues & Solutions

### ❌ "No shots found"
**Problem**: MongoDB doesn't have shot data  
**Solution**: Run Phase 2 first to populate the database

### ❌ "MongoDB Atlas client not configured"
**Problem**: Missing `MONGODB_ATLAS_URI` in `.env`  
**Solution**: Add your MongoDB connection string to `.env`

### ❌ "Google API key is required"
**Problem**: Missing `GOOGLE_API_KEY` in `.env`  
**Solution**: Add your Gemini API key to `.env`

### ⚠️ "Missing reference images"
**Problem**: `generated_images_s3` field is empty  
**Solution**: This is normal if you haven't uploaded images to S3 yet. The agent will still work.

## Next Steps

1. **Review Generated Prompts**: Check the output files and MongoDB
2. **Refine as Needed**: Adjust temperature or prompt structure if needed
3. **Upload to S3**: Add image URLs to `generated_images_s3` for better context
4. **Integrate with Video Generation**: Use the prompts with your video generation system

## API Endpoint Reference

**Endpoint**: `POST /image-video-generation`

**Request**:
```json
{
  "show_id": "string (required)",
  "episode_number": "integer (required)",
  "scene_description": "string (optional)"
}
```

**Response**:
```json
{
  "status": "success | error",
  "message": "string",
  "video_prompts_saved": "integer",
  "total_shots": "integer",
  "local_file": "string",
  "data_preview": [...]
}
```

## Configuration Options

Edit these in the agent initialization:

```python
VideoGenerationAgent(
    model_name="gemini-3.1-pro-preview",  # Gemini model to use
    temperature=0.7,                     # 0.0 = consistent, 1.0 = creative
    max_tokens=4096,                     # Maximum prompt length
    enable_saving=True,                  # Save local JSON files
    output_dir="phase_3_agents/output"  # Output directory
)
```

## Performance Tips

- **Parallel Processing**: For large episodes, consider batching
- **Caching**: Reuse MongoDB connection across multiple episodes
- **Monitoring**: Check logs for any API rate limiting
- **Testing**: Always test with a small episode first

## Support & Documentation

- **Full Documentation**: [README.md](./README.md)
- **API Docs**: http://localhost:8000/docs (when server is running)
- **Phase 2 Docs**: [../phase_2_agents/README.md](../phase_2_agents/README.md)

## Complete Workflow Example

```bash
# 1. Start the server
python api_server.py

# 2. In another terminal, test Phase 3
curl -X POST "http://localhost:8000/image-video-generation" \
  -H "Content-Type: application/json" \
  -d '{"show_id": "E01", "episode_number": 1}'

# 3. Check MongoDB
# (Use MongoDB Compass or CLI to view prompt_video_draft fields)

# 4. Check local output
cat phase_3_agents/output/E01_1_video_prompts_*.json | jq

# 5. Run tests
python test_video_generation_agent.py
```

## Using Postman 📮

For easier API testing, use the included Postman collection:

1. **Import the collection**
   ```
   File: postman_phase3_video_generation.json
   ```

2. **Available requests**:
   - ✅ Generate Video Prompts (Basic)
   - ✅ Generate Video Prompts with S3 URLs ⭐ Recommended
   - ✅ Generate Video Prompts - Complete Example
   - ✅ Health Check
   - ✅ Get Shots by Episode

3. **Quick test**:
   - Open "Generate Video Prompts with S3 URLs"
   - Update `shots_with_s3_images` with your S3 URLs
   - Click **Send**
   - Check response for `s3_urls_updated` and `video_prompts_saved`

See **[Postman Guide](../POSTMAN_PHASE3_GUIDE.md)** for detailed instructions.

---

## That's It! 🎉

You've successfully set up and run the Phase 3 Video Generation Agent. The agent will now generate comprehensive video prompts for all your shots based on all previous Phase 2 outputs, including S3 image references!

**Key Benefits of the S3 Input Approach**:
- ✅ Provide S3 URLs directly in the request
- ✅ Automatic storage in MongoDB `generated_images_s3` field
- ✅ Single API call for both S3 upload and video generation
- ✅ Easy to test with Postman

For more advanced usage, see the [full documentation](./README.md).

