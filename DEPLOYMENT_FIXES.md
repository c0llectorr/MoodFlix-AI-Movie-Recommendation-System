# Deployment Fixes for MoodFlix

## Issues Fixed

### 1. **503 Service Unavailable Error**
**Problem:** The container was crashing during initialization, returning 503 errors.

**Root Causes:**
- Model loading failures weren't being handled gracefully
- The app would crash if the model failed to load
- No proper startup state management

**Solution:**
- Modified the lifespan function to catch startup errors and set `app.state.model_ready = False` instead of crashing
- Added `app.state.startup_error` to track initialization errors
- App now starts even if model loading fails, allowing for debugging

### 2. **"Your space" is not valid JSON Error**
**Problem:** Frontend was receiving HTML error pages instead of JSON responses.

**Root Causes:**
- Hugging Face Spaces was returning HTML error pages when the container failed
- No proper error handling middleware to ensure JSON responses

**Solution:**
- Added `ModelReadinessMiddleware` to intercept requests and return proper JSON error responses
- Middleware checks if model is ready before processing requests
- Returns 503 with JSON error details if model is still initializing

### 3. **Network Issues - Could not resolve host: huggingface.co**
**Problem:** Docker container couldn't download models from Hugging Face during build.

**Root Causes:**
- Network connectivity issues in Hugging Face Spaces environment
- Model path issues causing download attempts

**Solution:**
- Changed model path from relative `./../../models/` to absolute `/app/models/`
- Models are now copied directly into the container during build
- No need for runtime downloads

### 4. **Model Path Issues**
**Problem:** Relative paths `./../../models/` don't work reliably in Docker.

**Solution:**
- Updated `Config.MODEL_PATH` to use absolute path `/app/models/`
- Dockerfile now properly copies models to `/app/models/`
- Works consistently across local and production environments

## Files Modified

### 1. **Dockerfile** (Complete Rewrite)
```dockerfile
- Changed base image to python:3.11-slim (more stable than 3.12)
- Added environment variables for caching (TORCH_HOME, HF_HOME, TRANSFORMERS_CACHE)
- Improved layer caching by copying requirements first
- Added startup script for better debugging
- Increased health check start period to 60s (model loading takes time)
- Added ca-certificates for SSL/TLS support
- Uses startup.sh script instead of direct uvicorn command
```

### 2. **app/backend/main.py** (Key Changes)
```python
# Model path fix
- Changed: MODEL_PATH = "./../../models/"
- To: MODEL_PATH = "/app/models/"

# Startup error handling
- Modified lifespan() to catch exceptions and set app.state.model_ready = False
- Added app.state.startup_error to track initialization errors
- App no longer crashes on model loading failure

# Health check improvement
- Added model readiness check
- Returns 503 if model is still initializing
- Provides error details in response

# Middleware addition
- Added ModelReadinessMiddleware to check model status before processing requests
- Returns proper JSON error responses instead of HTML
- Allows health check and root endpoints without model
```

### 3. **app/backend/requirements.txt**
```
- Added uvicorn[standard] for better production support
- Added starlette>=0.35.0 for middleware support
```

### 4. **app/backend/startup.sh** (New File)
```bash
- Startup script for better debugging
- Logs Python version, working directory, and available models
- Provides visibility into container initialization
```

### 5. **.dockerignore** (New File)
```
- Optimizes Docker build by excluding unnecessary files
- Reduces image size and build time
```

## How It Works Now

### Startup Flow:
1. Container starts
2. Startup script logs debug information
3. FastAPI app initializes with lifespan context manager
4. Model loading is attempted in the background
5. If model loads successfully: `app.state.model_ready = True`
6. If model fails to load: `app.state.model_ready = False` + error message stored
7. App is ready to receive requests

### Request Flow:
1. Request arrives at the API
2. `ModelReadinessMiddleware` checks if model is ready
3. If model not ready: Returns 503 JSON response with error details
4. If model ready: Request proceeds to the endpoint
5. Endpoint processes the request normally

### Health Check Flow:
1. Frontend calls `/health` endpoint
2. Health check verifies `app.state.model_ready`
3. If not ready: Returns 503 with initialization status
4. If ready: Returns 200 with healthy status
5. Frontend can retry or show loading state

## Deployment Instructions

### For Hugging Face Spaces:
1. Push the updated code to your repository
2. Hugging Face will automatically rebuild the Docker image
3. The container will start and begin model loading
4. Frontend will see 503 responses until model is ready
5. Once model loads, frontend will work normally

### For Local Testing:
```bash
# Build the image
docker build -t moodflix-api .

# Run the container
docker run -p 7860:7860 \
  -e TMDB_API_KEY="your_api_key" \
  moodflix-api

# Test the health endpoint
curl http://localhost:7860/health
```

## Monitoring

### Check Container Logs:
```bash
docker logs <container_id>
```

### Expected Log Output:
```
===== Application Startup at 2026-03-08 21:56:11 =====
Python version: Python 3.11.x
Working directory: /app
Model path: /app/models
Models available:
-rw-r--r-- 1 root root 710M ... classifier.pt
-rw-r--r-- 1 root root 2.0K ... config.json
...
Starting FastAPI application...
INFO:     Uvicorn running on http://0.0.0.0:7860
```

## Troubleshooting

### If you still see 503 errors:
1. Check container logs for model loading errors
2. Verify models are in `/app/models/` directory
3. Check if TMDB_API_KEY is set correctly
4. Increase health check start period if model takes longer to load

### If you see "Model not found" errors:
1. Verify models directory is copied in Dockerfile
2. Check model file permissions
3. Ensure model path is absolute: `/app/models/`

### If you see network errors:
1. Check internet connectivity in container
2. Verify TMDB_API_KEY is valid
3. Check firewall/proxy settings
