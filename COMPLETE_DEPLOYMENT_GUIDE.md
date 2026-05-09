# MoodFlix Complete Deployment Guide
## Backend on Hugging Face Spaces + Frontend on Vercel

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Part 1: Backend Deployment on Hugging Face Spaces](#part-1-backend-deployment-on-hugging-face-spaces)
4. [Part 2: Frontend Deployment on Vercel](#part-2-frontend-deployment-on-vercel)
5. [Part 3: Connecting Frontend to Backend](#part-3-connecting-frontend-to-backend)
6. [Verification & Testing](#verification--testing)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts:
- ✅ GitHub account (for version control)
- ✅ Hugging Face account (for backend deployment)
- ✅ Vercel account (for frontend deployment)
- ✅ TMDB API key (for movie recommendations)

### Required Tools (Local Machine):
- ✅ Git installed
- ✅ Node.js & npm (for frontend)
- ✅ Python 3.11+ (for backend testing)
- ✅ Docker (optional, for local testing)

### Get Your API Keys:
1. **TMDB API Key:**
   - Go to https://www.themoviedb.org/settings/api
   - Sign up/Login
   - Create an API key
   - Copy the key (you'll need it for Hugging Face)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐         ┌──────────────────────┐  │
│  │   FRONTEND (Vercel)  │         │  BACKEND (HF Spaces) │  │
│  │                      │         │                      │  │
│  │  - React App         │◄───────►│  - FastAPI Server    │  │
│  │  - Emotion Input     │ HTTPS   │  - DeBERTa Model     │  │
│  │  - Movie Display     │         │  - TMDB Integration  │  │
│  │                      │         │  - Model Files       │  │
│  │  URL:                │         │  URL:                │  │
│  │  moodflix.vercel.app │         │  mahmdshafee-...     │  │
│  │                      │         │  .hf.space           │  │
│  └──────────────────────┘         └──────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### What Goes Where:

**Hugging Face Spaces (Backend):**
- `Dockerfile`
- `app/backend/main.py`
- `app/backend/requirements.txt`
- `app/backend/startup.sh`
- `app/backend/.env` (with TMDB_API_KEY)
- `models/` directory (all model files)

**Vercel (Frontend):**
- `app/frontend/` directory (entire folder)
- `.env` file with backend URL

---

# PART 1: Backend Deployment on Hugging Face Spaces

## Step 1: Prepare Your GitHub Repository

### 1.1 Create a GitHub Repository (if you don't have one)

```bash
# Go to https://github.com/new
# Create a new repository named "MoodFlix-Backend"
# Choose: Public (so Hugging Face can access it)
# Do NOT initialize with README (we'll push existing code)
```

### 1.2 Initialize Git Locally (if not already done)

```bash
# Navigate to your project directory
cd "c:\Users\mahma\Desktop\MoodFlix-AI-Movie-Recommendation-System"

# Initialize git (if not already initialized)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: MoodFlix with fixed deployment"

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/MoodFlix-Backend.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 1.3 Verify Files on GitHub

Go to https://github.com/YOUR_USERNAME/MoodFlix-Backend and verify:
- ✅ `Dockerfile` exists in root
- ✅ `app/backend/` folder exists with all files
- ✅ `models/` folder exists with model files
- ✅ `.gitignore` is configured properly

---

## Step 2: Create Hugging Face Space

### 2.1 Go to Hugging Face Spaces

1. Visit https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in the form:
   - **Space name:** `emotion-detection-api` (or your preferred name)
   - **License:** Choose any (e.g., MIT)
   - **Space SDK:** Select "Docker"
   - **Visibility:** Public
4. Click "Create Space"

### 2.2 Connect Your GitHub Repository

After creating the space, you'll see options to set up the space:

**Option A: Using GitHub (Recommended)**
1. Click "Files and versions" tab
2. Click "Clone repository"
3. Copy the command shown
4. In your local terminal:
   ```bash
   cd "c:\Users\mahma\Desktop\MoodFlix-AI-Movie-Recommendation-System"
   git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api
   git push huggingface main
   ```

**Option B: Manual Upload**
1. Go to your Hugging Face Space
2. Click "Files and versions"
3. Click "Add file" → "Upload files"
4. Upload all necessary files (see Step 3 below)

---

## Step 3: Upload Backend Files to Hugging Face

### 3.1 Files to Upload (Complete List)

Create this exact structure in your Hugging Face Space:

```
emotion-detection-api/
├── Dockerfile                    ← Root level
├── app/
│   └── backend/
│       ├── main.py              ← Main FastAPI app
│       ├── requirements.txt      ← Python dependencies
│       ├── startup.sh            ← Startup script
│       └── .env                  ← Environment variables
└── models/
    ├── classifier.pt            ← Model weights
    ├── config.json              ← Model config
    ├── model.safetensors        ← Model safetensors
    └── metrics.json             ← Model metrics
```

### 3.2 Dockerfile (Already Prepared)

**Location:** Root of repository
**File:** `Dockerfile`
**Status:** ✅ Already created and fixed

The Dockerfile is already in your project. It will:
- Use Python 3.11-slim
- Install all dependencies from requirements.txt
- Copy models from `models/` directory
- Copy backend code from `app/backend/`
- Run the startup script

### 3.3 Backend Files

**Location:** `app/backend/`

**Files needed:**
1. **main.py** - FastAPI application
   - Status: ✅ Already fixed
   - Contains: Emotion detection, movie recommendations, CORS setup

2. **requirements.txt** - Python dependencies
   - Status: ✅ Already updated
   - Contains: fastapi, torch, transformers, etc.

3. **startup.sh** - Startup script
   - Status: ✅ Already created
   - Contains: Debug logging and uvicorn startup

4. **.env** - Environment variables
   - Status: ⚠️ NEEDS YOUR TMDB API KEY
   - Content:
     ```
     TMDB_API_KEY="your_actual_api_key_here"
     ```

### 3.4 Model Files

**Location:** `models/`

**Files needed:**
1. **classifier.pt** (710MB)
   - Status: ✅ Already in your project
   - Contains: Trained model weights

2. **config.json** (2KB)
   - Status: ✅ Already in your project
   - Contains: Model configuration

3. **model.safetensors** (710MB)
   - Status: ✅ Already in your project
   - Contains: Model in safetensors format

4. **metrics.json** (1KB)
   - Status: ✅ Already in your project
   - Contains: Model performance metrics

---

## Step 4: Configure Environment Variables on Hugging Face

### 4.1 Add TMDB API Key to Hugging Face Space

1. Go to your Hugging Face Space: https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api
2. Click "Settings" (gear icon)
3. Scroll to "Repository secrets"
4. Click "Add a secret"
5. Add:
   - **Name:** `TMDB_API_KEY`
   - **Value:** Your actual TMDB API key (from prerequisites)
6. Click "Add secret"

### 4.2 Update .env File in Repository

In your local `app/backend/.env`:
```
TMDB_API_KEY="your_actual_api_key_here"
```

Then push to GitHub:
```bash
git add app/backend/.env
git commit -m "Add TMDB API key"
git push origin main
git push huggingface main
```

---

## Step 5: Deploy Backend to Hugging Face

### 5.1 Push Code to Hugging Face

```bash
# From your project directory
cd "c:\Users\mahma\Desktop\MoodFlix-AI-Movie-Recommendation-System"

# Add Hugging Face remote (if not already added)
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api

# Push to Hugging Face
git push huggingface main
```

### 5.2 Monitor Deployment

1. Go to your Hugging Face Space
2. Click "Logs" tab
3. Watch the build process:
   - Building Docker image
   - Installing dependencies
   - Loading model
   - Starting application

**Expected output:**
```
===== Application Startup at 2026-03-08 21:56:11 =====
Python version: Python 3.11.x
Working directory: /app
Model path: /app/models
Models available:
-rw-r--r-- 1 root root 710M ... classifier.pt
Starting FastAPI application...
INFO:     Uvicorn running on http://0.0.0.0:7860
```

### 5.3 Wait for Deployment to Complete

- ⏱️ Build time: 5-10 minutes (first time)
- ⏱️ Model loading: 1-2 minutes
- ⏱️ Total: 10-15 minutes

**Status indicators:**
- 🟡 Yellow: Building/Loading
- 🟢 Green: Running successfully
- 🔴 Red: Error (check logs)

### 5.4 Get Your Backend URL

Once deployed successfully:
1. Go to your Hugging Face Space
2. Copy the URL from the top (e.g., `https://mahmdshafee-emotion-detection-api.hf.space`)
3. Save this URL - you'll need it for the frontend

---

## Step 6: Test Backend Deployment

### 6.1 Test Health Endpoint

```bash
# Replace with your actual Hugging Face Space URL
curl https://mahmdshafee-emotion-detection-api.hf.space/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "model_loaded": true,
  "memory_mb": 2048.5
}
```

### 6.2 Test Recommendations Endpoint

```bash
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling amazing today!"}'
```

**Expected response:**
```json
{
  "emotion": "joy",
  "confidence": 0.95,
  "recommendations": [
    {
      "genre": "Comedy",
      "movies": [
        {
          "id": 278,
          "title": "The Shawshank Redemption",
          "poster_path": "/...",
          "vote_average": 8.7,
          "release_date": "1994-09-23"
        }
      ]
    }
  ]
}
```

### 6.3 Verify CORS is Working

The backend should accept requests from any origin (configured in main.py).

---

# PART 2: Frontend Deployment on Vercel

## Step 1: Prepare Frontend Files

### 1.1 Update Frontend Environment Variables

**File:** `app/frontend/.env`

```
VITE_API_BASE_URL="https://mahmdshafee-emotion-detection-api.hf.space"
```

Replace with your actual Hugging Face Space URL.

### 1.2 Verify Frontend Structure

```
app/frontend/
├── src/
│   ├── App.jsx              ← Main component (already fixed)
│   ├── main.jsx
│   ├── index.css
│   └── components/
│       ├── MovieCard.jsx
│       └── MovieCarousel.jsx
├── public/
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── index.html
└── .env                     ← Update with backend URL
```

### 1.3 Commit Frontend Changes

```bash
cd "c:\Users\mahma\Desktop\MoodFlix-AI-Movie-Recommendation-System"

# Update .env with your backend URL
# Edit: app/frontend/.env
# Change VITE_API_BASE_URL to your Hugging Face Space URL

# Commit changes
git add app/frontend/.env
git commit -m "Update backend URL for production"
git push origin main
```

---

## Step 2: Create Vercel Project

### 2.1 Go to Vercel

1. Visit https://vercel.com
2. Click "Sign up" or "Log in"
3. Choose "Continue with GitHub"
4. Authorize Vercel to access your GitHub account

### 2.2 Import Project

1. Click "Add New..." → "Project"
2. Click "Import Git Repository"
3. Paste your GitHub repository URL:
   ```
   https://github.com/YOUR_USERNAME/MoodFlix-Backend
   ```
4. Click "Import"

### 2.3 Configure Project Settings

**Framework Preset:** Select "Vite"

**Root Directory:** 
- Click "Edit"
- Change to: `app/frontend`
- Click "Save"

**Environment Variables:**
1. Click "Environment Variables"
2. Add:
   - **Name:** `VITE_API_BASE_URL`
   - **Value:** `https://mahmdshafee-emotion-detection-api.hf.space`
   - **Environments:** Production, Preview, Development
3. Click "Add"

### 2.4 Deploy

1. Click "Deploy"
2. Wait for deployment to complete (2-5 minutes)
3. You'll see a success message with your Vercel URL

**Your frontend URL will be:** `https://moodflix.vercel.app` (or similar)

---

## Step 3: Configure Vercel Build Settings

### 3.1 Build Command

**Default:** `npm run build`

This should work automatically for Vite projects.

### 3.2 Output Directory

**Default:** `dist`

This is correct for Vite.

### 3.3 Install Command

**Default:** `npm install`

This is correct.

---

## Step 4: Test Frontend Deployment

### 4.1 Visit Your Frontend URL

Go to: `https://moodflix.vercel.app` (or your Vercel URL)

### 4.2 Test the Application

1. Type some text in the input box
2. Click "Detect & Suggest"
3. Wait for emotion detection
4. Verify movies are displayed

### 4.3 Check Browser Console

Open Developer Tools (F12):
- **Console tab:** Should show no errors
- **Network tab:** Should show successful requests to your backend

---

# PART 3: Connecting Frontend to Backend

## Step 1: Verify CORS Configuration

### 1.1 Backend CORS Settings

**File:** `app/backend/main.py` (lines ~387-398)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",           # Local development
        "http://localhost:8000",           # Local backend
        "https://moodflix.vercel.app",     # Your Vercel frontend
        "https://*.vercel.app",            # All Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 1.2 Update CORS for Your Frontend URL

If your Vercel URL is different:

```bash
# Edit app/backend/main.py
# Find the CORS configuration
# Add your Vercel URL to allow_origins list
# Example: "https://your-custom-domain.vercel.app"

# Commit and push
git add app/backend/main.py
git commit -m "Update CORS for Vercel frontend"
git push origin main
git push huggingface main
```

## Step 2: Verify Frontend Configuration

### 2.1 Frontend API URL

**File:** `app/frontend/src/App.jsx` (line ~13)

```javascript
const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
```

This automatically uses the environment variable from `.env`.

### 2.2 Verify .env File

**File:** `app/frontend/.env`

```
VITE_API_BASE_URL="https://mahmdshafee-emotion-detection-api.hf.space"
```

---

## Step 3: Test Connection

### 3.1 Test from Frontend

1. Go to your Vercel frontend URL
2. Open Developer Tools (F12)
3. Go to "Network" tab
4. Type text and submit
5. Look for POST request to `/recommendations`
6. Verify response status is 200 (not 503 or 404)

### 3.2 Test from Backend

```bash
# Test backend directly
curl https://mahmdshafee-emotion-detection-api.hf.space/health

# Should return:
# {"status":"healthy","device":"cpu","model_loaded":true,"memory_mb":2048.5}
```

### 3.3 Test Full Flow

1. Frontend: Type "I am feeling happy"
2. Frontend: Click "Detect & Suggest"
3. Backend: Receives request
4. Backend: Detects emotion
5. Backend: Fetches movies from TMDB
6. Frontend: Displays results

---

# Verification & Testing

## Complete Checklist

### Backend (Hugging Face)
- [ ] Hugging Face Space created
- [ ] Files uploaded (Dockerfile, app/backend/, models/)
- [ ] TMDB_API_KEY set in secrets
- [ ] Deployment completed (green status)
- [ ] Health endpoint returns 200
- [ ] Recommendations endpoint returns movies
- [ ] CORS headers present in responses

### Frontend (Vercel)
- [ ] Vercel project created
- [ ] Root directory set to `app/frontend`
- [ ] VITE_API_BASE_URL environment variable set
- [ ] Deployment completed
- [ ] Frontend loads without errors
- [ ] Can type text and submit
- [ ] Receives emotion detection results
- [ ] Displays movie recommendations

### Integration
- [ ] Frontend can reach backend
- [ ] No CORS errors in console
- [ ] Movies display correctly
- [ ] All features work end-to-end

---

## Testing Scenarios

### Scenario 1: Emotion Detection
```
Input: "I am feeling amazing today!"
Expected: 
- Emotion: joy
- Confidence: >0.8
- Movies: Comedy, Adventure, Family, Animation genres
```

### Scenario 2: Error Handling
```
Input: "" (empty)
Expected: Error message "Please enter some text"
```

### Scenario 3: Long Text
```
Input: 5000+ characters
Expected: Error message "Text too long (max 5000 characters)"
```

### Scenario 4: Network Error
```
Scenario: Backend is down
Expected: Error message "Failed to connect to server"
```

---

# Troubleshooting

## Backend Issues

### Issue: 503 Service Unavailable
**Cause:** Model still loading or failed to load
**Solution:**
1. Check Hugging Face Space logs
2. Wait 2-3 minutes for model to load
3. Verify model files exist in `/app/models/`
4. Check TMDB_API_KEY is set correctly

### Issue: "Could not resolve host: huggingface.co"
**Cause:** Network connectivity issue during build
**Solution:**
1. Rebuild the space (click "Restart" in settings)
2. Check internet connectivity
3. Verify models are already in the repository

### Issue: "Model not found"
**Cause:** Model files not copied to container
**Solution:**
1. Verify `models/` directory exists in repository
2. Check Dockerfile copies models correctly
3. Rebuild the space

### Issue: TMDB movies not fetching
**Cause:** Invalid or missing TMDB_API_KEY
**Solution:**
1. Verify TMDB_API_KEY is set in Hugging Face secrets
2. Test API key: https://www.themoviedb.org/settings/api
3. Rebuild the space after updating secret

---

## Frontend Issues

### Issue: "Failed to fetch" error
**Cause:** Frontend can't reach backend
**Solution:**
1. Verify backend URL in `.env` is correct
2. Check backend is running (test health endpoint)
3. Verify CORS is configured correctly
4. Check browser console for exact error

### Issue: Movies not displaying
**Cause:** Backend not returning movies
**Solution:**
1. Check backend logs for errors
2. Verify TMDB_API_KEY is valid
3. Test backend directly with curl
4. Check network tab in browser for response

### Issue: Blank page or 404
**Cause:** Frontend not deployed correctly
**Solution:**
1. Check Vercel deployment logs
2. Verify root directory is `app/frontend`
3. Check build command is `npm run build`
4. Verify all dependencies installed

---

## Common Errors & Solutions

### Error: "VITE_API_BASE_URL is undefined"
**Solution:** 
- Add `.env` file to `app/frontend/`
- Set `VITE_API_BASE_URL="your_backend_url"`
- Rebuild on Vercel

### Error: "CORS error: Access-Control-Allow-Origin"
**Solution:**
- Add your Vercel URL to CORS allow_origins in main.py
- Rebuild backend on Hugging Face

### Error: "TypeError: Failed to fetch"
**Solution:**
- Check backend URL is correct
- Verify backend is running
- Check network connectivity
- Look at browser console for details

### Error: "SyntaxError: Unexpected token 'Y', 'Your space' is not valid JSON"
**Solution:**
- Backend is returning HTML error page
- Check backend logs
- Verify model loaded successfully
- Wait for initialization to complete

---

## Performance Optimization

### Backend Performance
- Model loading: 1-2 minutes (first time)
- Emotion detection: 100-200ms
- Movie fetching: 500-1000ms
- Total response: 1-2 seconds

### Frontend Performance
- Page load: <2 seconds
- Emotion detection: 1-2 seconds
- Movie display: Instant

### Optimization Tips
1. **Backend:** Use CPU inference (no GPU on free tier)
2. **Frontend:** Lazy load movie images
3. **Both:** Enable caching headers
4. **Network:** Use CDN (Vercel provides this)

---

## Monitoring & Maintenance

### Check Backend Status
```bash
curl https://mahmdshafee-emotion-detection-api.hf.space/health
```

### View Backend Logs
1. Go to Hugging Face Space
2. Click "Logs" tab
3. Scroll to see recent activity

### View Frontend Logs
1. Go to Vercel project
2. Click "Deployments"
3. Click latest deployment
4. Click "Logs"

### Update Backend
```bash
# Make changes locally
git add .
git commit -m "Update backend"
git push origin main
git push huggingface main
# Hugging Face will auto-rebuild
```

### Update Frontend
```bash
# Make changes locally
git add .
git commit -m "Update frontend"
git push origin main
# Vercel will auto-rebuild
```

---

## Final Checklist Before Going Live

- [ ] Backend deployed on Hugging Face (green status)
- [ ] Frontend deployed on Vercel (green status)
- [ ] TMDB_API_KEY configured on Hugging Face
- [ ] Backend URL in frontend `.env`
- [ ] CORS configured for frontend URL
- [ ] Health endpoint returns 200
- [ ] Recommendations endpoint returns movies
- [ ] Frontend loads without errors
- [ ] Can detect emotions
- [ ] Can display movies
- [ ] No console errors
- [ ] Tested on mobile (responsive)
- [ ] Tested with different emotions
- [ ] Tested error scenarios

---

## Quick Reference URLs

### Your Deployed Services:
- **Backend:** `https://mahmdshafee-emotion-detection-api.hf.space`
- **Frontend:** `https://moodflix.vercel.app`
- **GitHub:** `https://github.com/YOUR_USERNAME/MoodFlix-Backend`
- **Hugging Face Space:** `https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api`
- **Vercel Project:** `https://vercel.com/dashboard`

### API Endpoints:
- **Health:** `GET /health`
- **Emotions:** `GET /emotions`
- **Predict:** `POST /predict`
- **Recommendations:** `POST /recommendations`
- **Batch Predict:** `POST /batch_predict`

### Documentation:
- **FastAPI Docs:** `https://mahmdshafee-emotion-detection-api.hf.space/docs`
- **OpenAPI Schema:** `https://mahmdshafee-emotion-detection-api.hf.space/openapi.json`

---

## Support & Resources

### If Something Goes Wrong:
1. Check the logs (Hugging Face or Vercel)
2. Review the troubleshooting section above
3. Test endpoints individually with curl
4. Check browser console for errors
5. Verify environment variables are set

### Useful Commands:

**Test backend health:**
```bash
curl https://mahmdshafee-emotion-detection-api.hf.space/health
```

**Test emotion detection:**
```bash
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am happy"}'
```

**Test recommendations:**
```bash
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling amazing!"}'
```

---

## Next Steps After Deployment

1. **Monitor Performance:** Check logs regularly
2. **Gather Feedback:** Test with users
3. **Optimize:** Improve based on feedback
4. **Scale:** Consider paid tiers if needed
5. **Maintain:** Keep dependencies updated

---

**Last Updated:** March 8, 2026
**Status:** ✅ Ready for Deployment
**Estimated Deployment Time:** 20-30 minutes total

---

## Quick Start Summary

### For Impatient Users:

1. **Backend (10 minutes):**
   ```bash
   # Push to GitHub
   git push origin main
   
   # Create Hugging Face Space with Docker
   # Push to Hugging Face
   git push huggingface main
   
   # Wait for deployment (5-10 minutes)
   ```

2. **Frontend (5 minutes):**
   ```bash
   # Update app/frontend/.env with backend URL
   # Push to GitHub
   git push origin main
   
   # Import project in Vercel
   # Deploy (2-5 minutes)
   ```

3. **Test (5 minutes):**
   ```bash
   # Visit frontend URL
   # Type text and submit
   # Verify movies display
   ```

**Total Time:** ~20-30 minutes

---

**Congratulations! Your MoodFlix app is now deployed! 🎉**
