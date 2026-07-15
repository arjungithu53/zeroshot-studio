# Agent 3 - Prompt Review Agent: Quick Start Guide

## What is Agent 3?

Agent 3 automatically reviews AI-generated image prompts to ensure visual and narrative continuity across your shot sequence. It catches common errors like:

- 🌅 Sunset changing to sunrise between shots
- 👔 Character costumes changing unexpectedly
- ☀️ Lighting direction inconsistencies
- 🌧️ Weather condition mismatches
- 📍 Character position/blocking errors

## How to Use

### Method 1: Automatic (Recommended)

Agent 3 runs automatically when you call the main API endpoint:

```bash
POST http://localhost:8000/analyze-shots-mongodb?show_id=SHOW123&episode_number=1

# Agent 1 runs → Determines strategies
# Agent 2 runs → Generates prompts
# Agent 3 runs → Reviews prompts ✨ AUTOMATIC
```

### Method 2: Standalone Testing

Test the agent independently:

```bash
python test_prompt_review_agent.py
```

This runs example scenarios with intentional continuity errors to show how the agent detects and fixes them.

## What You Get

### 1. MongoDB Updates
Each shot gets a new field:
```javascript
{
  "prompt_image_draft": "Original prompt from Agent 2...",
  "prompt_image_reviewed": "Refined prompt with continuity fixes..." ✨
}
```

### 2. Review Report
Detailed JSON file saved to `outputs/`:
```json
{
  "shot_id": "S01E01_002",
  "original_prompt": "Close-up of door in morning sunlight...",
  "reviewed_prompt": "Close-up of door in golden sunset light...",
  "changes_made": [
    "Changed 'morning sunlight' to 'golden sunset light' for continuity with Shot 1"
  ],
  "shot_modified": true,
  "reason_for_modification": "Fixed time-of-day inconsistency",
  "continuity_observations": [
    "Now matches sunset lighting from establishing shot"
  ]
}
```

## Example: Detecting Errors

**Shot 1** (Original):
```
"Wide shot of lakeside cottage at golden hour sunset..."
```

**Shot 2** (Original - ERROR):
```
"Close-up of door in bright morning sunlight..."
```
❌ Problem: Changed from sunset to morning!

**Shot 2** (After Review - FIXED):
```
"Close-up of door in warm golden hour sunset light..."
```
✅ Fixed: Now consistent with Shot 1

## Configuration

Set these in your `.env` file:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
MONGODB_ATLAS_URI=mongodb+srv://...
```

## Check Results

### Via API Response
```json
{
  "prompt_review_response": {
    "success": true,
    "message": "Successfully reviewed prompts for 10 shots",
    "total_shots": 10,
    "shots_modified": 3
  }
}
```

### Via MongoDB
Query the shots collection:
```javascript
db.shots.find({
  "show_id": "SHOW123",
  "episode_number": 1
})

// Each shot now has:
// - prompt_image_draft (from Agent 2)
// - prompt_image_reviewed (from Agent 3) ✨
```

### Via Local Files
Check the outputs directory:
```bash
ls phase_2_agents/agent_prompt_review/outputs/
# review_E01_20251008_123456.json
```

## When Does It Run?

Agent 3 automatically runs **after Agent 2** completes:

```
/analyze-shots-mongodb
  ├─ Agent 1: Analyze strategies ✓
  ├─ Agent 2: Generate prompts ✓
  └─ Agent 3: Review prompts ✓ ← AUTOMATICALLY
```

You don't need to call it separately!

## What It Checks

✅ **Lighting**: Direction, quality, color temperature  
✅ **Weather**: Rain, fog, sun, clouds  
✅ **Time of Day**: Morning, sunset, night  
✅ **Characters**: Costume, position, appearance  
✅ **Spatial Logic**: Blocking, screen direction  
✅ **Colors**: Palette consistency  
✅ **Atmosphere**: Mood and environmental effects  

## Minimal Edits Philosophy

The agent only fixes **actual continuity problems**. It won't:
- Rewrite prompts unnecessarily
- Change artistic style
- Add unrelated details
- Override Agent 2's creative work

It makes **surgical fixes** to ensure continuity.

## Need Help?

- 📖 Full docs: `phase_2_agents/agent_prompt_review/README.md`
- 🧪 Run tests: `python test_prompt_review_agent.py`
- 📝 Implementation details: `AGENT_3_IMPLEMENTATION_SUMMARY.md`

---

**Agent 3 is ready to ensure your shots have perfect visual continuity!** 🎬✨

