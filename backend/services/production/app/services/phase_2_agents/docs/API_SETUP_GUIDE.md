# Shot Strategy Agent API Setup Guide

This guide explains how to set up and use the Shot Strategy Agent API with Postman testing.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Copy the example environment file and configure it:
```bash
cp env.example .env
```

Edit `.env` with your actual values:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for MongoDB Atlas integration)
MONGODB_ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE_NAME=production
MONGODB_SHOTS_COLLECTION=shots
```

### 3. Start the Server
```bash
python3 start_server.py
```

The server will start on `http://localhost:8000`

### 4. Test the API
```bash
python3 test_api.py
```

## 📡 API Endpoints

### Health Check
- **GET** `/health`
- Returns server status and configuration

### Analyze Shots
- **POST** `/analyze-shots`
- Analyzes shot list and returns generation strategies
- **Body**: JSON with episode_id, title, and shots array

### Analyze Shots with MongoDB
- **POST** `/analyze-shots-mongodb?show_id={show_id}&episode_number={episode_number}`
- Same as above but saves results to MongoDB Atlas

### MongoDB Operations
- **GET** `/mongodb/stats` - Get collection statistics
- **GET** `/mongodb/shots/{show_id}/{episode_number}` - Retrieve shots

## 🧪 Postman Testing

### Import Collection
1. Open Postman
2. Click "Import" 
3. Select `postman_collection.json`
4. Set environment variable `base_url` to `http://localhost:8000`

### Test Requests

#### 1. Health Check
- **Method**: GET
- **URL**: `{{base_url}}/health`
- **Expected**: 200 OK with server status

#### 2. Analyze Shots
- **Method**: POST
- **URL**: `{{base_url}}/analyze-shots`
- **Body**: JSON with shot list (see example in collection)

#### 3. Analyze with MongoDB
- **Method**: POST
- **URL**: `{{base_url}}/analyze-shots-mongodb?show_id=507f1f77bcf86cd799439010&episode_number=1`
- **Body**: JSON with shot list
- **Expected**: Analysis results + MongoDB save confirmation

## 📝 Example Request Body

```json
{
  "episode_id": "E01",
  "title": "The Beginning",
  "shots": [
    {
      "shot_id": "S01E01_001",
      "description": "Wide establishing shot of a bustling city street at dawn",
      "duration": 5.0,
      "scene_number": 1,
      "sequence_number": 1,
      "shot_style": "wide_shot",
      "camera_movement": "pan",
      "source_type": "generated",
      "generated_image_id": "507f1f77bcf86cd799439011"
    },
    {
      "shot_id": "S01E01_002",
      "description": "Close-up of protagonist's determined face",
      "duration": 3.5,
      "scene_number": 1,
      "sequence_number": 2,
      "shot_style": "close_up",
      "camera_movement": "push_in",
      "source_type": "generated",
      "generated_image_id": "507f1f77bcf86cd799439012"
    }
  ]
}
```

## 📊 Example Response

```json
{
  "episode_id": "E01",
  "title": "The Beginning",
  "annotated_shots": [
    {
      "shot_id": "S01E01_001",
      "description": "Wide establishing shot of a bustling city street at dawn",
      "generation_strategy": "generate_new",
      "reasoning": "First shot of the sequence, no previous context available",
      "confidence_score": 0.95,
      "continuity_notes": "This is the first shot of the sequence"
    },
    {
      "shot_id": "S01E01_002",
      "description": "Close-up of protagonist's determined face",
      "generation_strategy": "last_frame_seed",
      "reasoning": "Character continuity with previous shot, using last frame as seed",
      "confidence_score": 0.85,
      "continuity_notes": "Character continuity, camera movement from wide to close-up"
    }
  ],
  "strategy_summary": {
    "generate_new": 1,
    "last_frame_seed": 1,
    "multi_shot": 0
  },
  "overall_continuity_notes": "Strong continuity detected across 1 shot transitions"
}
```

## 🔧 Configuration Options

### Environment Variables
- `OPENAI_API_KEY`: Required - Your OpenAI API key
- `MONGODB_ATLAS_URI`: Optional - MongoDB Atlas connection string
- `MONGODB_DATABASE_NAME`: Optional - Database name (default: production)
- `MONGODB_SHOTS_COLLECTION`: Optional - Collection name (default: shots)
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `API_DEBUG`: Debug mode (default: True)

### Server Configuration
The server can be configured via environment variables or by modifying `start_server.py`.

## 🐛 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Set your OpenAI API key in the `.env` file
   - Make sure the `.env` file is in the project root

2. **"MongoDB Atlas client not configured"**
   - This is normal if you haven't set up MongoDB Atlas
   - The API will work without MongoDB, just won't save results

3. **"Connection refused"**
   - Make sure the server is running on the correct port
   - Check if another process is using port 8000

4. **"Import error"**
   - Run `pip install -r requirements.txt` to install dependencies

### Debug Mode
Set `API_DEBUG=True` in your `.env` file for detailed logging and auto-reload.

### Logs
Check the console output for detailed error messages and API logs.

## 🚀 Production Deployment

### Environment Setup
1. Set `API_DEBUG=False` for production
2. Configure proper CORS origins
3. Use environment-specific MongoDB Atlas clusters
4. Set up proper logging and monitoring

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "start_server.py"]
```

### Health Monitoring
Use the `/health` endpoint for health checks and monitoring.

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🎯 Next Steps

1. **Test with Postman**: Import the collection and run all tests
2. **Set up MongoDB Atlas**: Configure your database for full functionality
3. **Customize prompts**: Modify the agent prompts for your specific use case
4. **Scale up**: Deploy to production with proper monitoring

## 📞 Support

For issues or questions:
1. Check the logs for error messages
2. Verify your environment variables
3. Test individual endpoints with Postman
4. Review the API documentation at `/docs`
