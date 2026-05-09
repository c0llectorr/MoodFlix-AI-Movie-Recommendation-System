# MoodFlix Deployment Visual Guide

## 📊 Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOUR MOODFLIX APPLICATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        USER'S BROWSER                                │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Frontend (React + Vite)                                       │ │   │
│  │  │  https://moodflix.vercel.app                                   │ │   │
│  │  │                                                                │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐ │ │   │
│  │  │  │ Input: "I am feeling amazing!"                           │ │ │   │
│  │  │  │ Button: "Detect & Suggest"                              │ │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘ │ │   │
│  │  │                          ↓                                     │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐ │ │   │
│  │  │  │ Output: Emotion + Movies                                 │ │ │   │
│  │  │  │ - Emotion: Joy (95% confidence)                          │ │ │   │
│  │  │  │ - Movies: Comedy, Adventure, Family, Animation           │ │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘ │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  ↕                                           │
│                            HTTPS Request/Response                            │
│                                  ↕                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    BACKEND (Hugging Face Spaces)                     │   │
│  │  https://mahmdshafee-emotion-detection-api.hf.space                 │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ FastAPI Server (Python)                                        │ │   │
│  │  │                                                                │ │   │
│  │  │ POST /recommendations                                         │ │   │
│  │  │ ├─ Input: {"text": "I am feeling amazing!"}                  │ │   │
│  │  │ ├─ Process:                                                  │ │   │
│  │  │ │  1. Load DeBERTa Model                                    │ │   │
│  │  │ │  2. Tokenize text                                         │ │   │
│  │  │ │  3. Run inference                                         │ │   │
│  │  │ │  4. Get emotion probabilities                             │ │   │
│  │  │ │  5. Map emotion to genres                                 │ │   │
│  │  │ │  6. Fetch movies from TMDB                                │ │   │
│  │  │ └─ Output: {"emotion": "joy", "recommendations": [...]}     │ │   │
│  │  │                                                                │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ DeBERTa Model (710MB)                                          │ │   │
│  │  │ - Trained on emotion classification                            │ │   │
│  │  │ - 7 emotion classes: anger, fear, joy, love, neutral,         │ │   │
│  │  │   sadness, surprise                                            │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ TMDB API Integration                                           │ │   │
│  │  │ - Fetch movies by genre                                        │ │   │
│  │  │ - Get movie details (title, poster, rating, etc.)             │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  ↕                                           │
│                            HTTPS Request                                     │
│                                  ↕                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    EXTERNAL API (TMDB)                               │   │
│  │  https://api.themoviedb.org/3                                       │   │
│  │                                                                      │   │
│  │  - Movie database                                                   │   │
│  │  - Genre information                                                │   │
│  │  - Movie details and posters                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

1. USER ENTERS TEXT
   ┌──────────────────┐
   │ "I am so happy!" │
   └────────┬─────────┘
            │
            ↓
2. FRONTEND SENDS REQUEST
   ┌────────────────────────────────────────┐
   │ POST /recommendations                  │
   │ Content-Type: application/json         │
   │ Body: {"text": "I am so happy!"}       │
   └────────┬─────────────────────────────────┘
            │
            ↓ (HTTPS)
3. BACKEND RECEIVES REQUEST
   ┌────────────────────────────────────────┐
   │ FastAPI Server                         │
   │ Validates input                        │
   │ Checks model is ready                  │
   └────────┬─────────────────────────────────┘
            │
            ↓
4. EMOTION DETECTION
   ┌────────────────────────────────────────┐
   │ DeBERTa Model                          │
   │ Input: "I am so happy!"                │
   │ Output: {                              │
   │   "emotion": "joy",                    │
   │   "confidence": 0.95,                  │
   │   "probabilities": {...}               │
   │ }                                      │
   └────────┬─────────────────────────────────┘
            │
            ↓
5. MAP EMOTION TO GENRES
   ┌────────────────────────────────────────┐
   │ joy → [Comedy, Adventure, Family,      │
   │        Animation]                      │
   └────────┬─────────────────────────────────┘
            │
            ↓
6. FETCH MOVIES FROM TMDB
   ┌────────────────────────────────────────┐
   │ For each genre:                        │
   │ - Get genre ID from TMDB               │
   │ - Fetch 12 popular movies              │
   │ - Extract: title, poster, rating, etc. │
   └────────┬─────────────────────────────────┘
            │
            ↓
7. FORMAT RESPONSE
   ┌────────────────────────────────────────┐
   │ {                                      │
   │   "emotion": "joy",                    │
   │   "confidence": 0.95,                  │
   │   "recommendations": [                 │
   │     {                                  │
   │       "genre": "Comedy",               │
   │       "movies": [                      │
   │         {                              │
   │           "id": 278,                   │
   │           "title": "Movie Name",       │
   │           "poster_path": "/...",       │
   │           "vote_average": 8.7,         │
   │           "release_date": "2024-01-01" │
   │         }                              │
   │       ]                                │
   │     }                                  │
   │   ]                                    │
   │ }                                      │
   └────────┬─────────────────────────────────┘
            │
            ↓ (HTTPS)
8. FRONTEND RECEIVES RESPONSE
   ┌────────────────────────────────────────┐
   │ Status: 200 OK                         │
   │ Body: JSON with emotion and movies     │
   └────────┬─────────────────────────────────┘
            │
            ↓
9. DISPLAY RESULTS
   ┌────────────────────────────────────────┐
   │ Show emotion with icon and confidence  │
   │ Display movie carousels by genre       │
   │ Show movie posters and details         │
   └────────────────────────────────────────┘
```

---

## 📦 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          YOUR LOCAL MACHINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Project Directory                                                    │   │
│  │ MoodFlix-AI-Movie-Recommendation-System/                             │   │
│  │                                                                      │   │
│  │ ├── Dockerfile                    ← Copied to HF                    │   │
│  │ ├── app/                                                             │   │
│  │ │   ├── backend/                  ← Copied to HF                    │   │
│  │ │   │   ├── main.py                                                 │   │
│  │ │   │   ├── requirements.txt                                        │   │
│  │ │   │   ├── startup.sh                                              │   │
│  │ │   │   └── .env                                                    │   │
│  │ │   └── frontend/                 ← Copied to Vercel               │   │
│  │ │       ├── src/                                                    │   │
│  │ │       ├── public/                                                 │   │
│  │ │       ├── package.json                                            │   │
│  │ │       ├── vite.config.js                                          │   │
│  │ │       └── .env                                                    │   │
│  │ └── models/                       ← Copied to HF                    │   │
│  │     ├── classifier.pt                                               │   │
│  │     ├── config.json                                                 │   │
│  │     ├── model.safetensors                                           │   │
│  │     └── metrics.json                                                │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                            git push origin main
                                    ↓
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ↓                                                       ↓
┌──────────────────────────────┐                    ┌──────────────────────────────┐
│    GITHUB REPOSITORY         │                    │    GITHUB REPOSITORY         │
│  (MoodFlix-Backend)          │                    │  (MoodFlix-Backend)          │
│                              │                    │                              │
│ ├── Dockerfile               │                    │ ├── Dockerfile               │
│ ├── app/backend/             │                    │ ├── app/frontend/            │
│ ├── models/                  │                    │ └── app/backend/             │
│ └── ...                      │                    │                              │
└──────────────────────────────┘                    └──────────────────────────────┘
        ↓                                                       ↓
   git push huggingface main                          Import in Vercel
        ↓                                                       ↓
┌──────────────────────────────┐                    ┌──────────────────────────────┐
│  HUGGING FACE SPACES         │                    │    VERCEL PROJECT            │
│  (emotion-detection-api)     │                    │  (moodflix)                  │
│                              │                    │                              │
│ ✅ Docker Build              │                    │ ✅ npm install               │
│ ✅ Install Dependencies      │                    │ ✅ npm run build             │
│ ✅ Copy Models               │                    │ ✅ Deploy to CDN             │
│ ✅ Load Model                │                    │                              │
│ ✅ Start FastAPI             │                    │ 🌐 https://moodflix.         │
│                              │                    │    vercel.app                │
│ 🌐 https://mahmdshafee-     │                    │                              │
│    emotion-detection-api.    │                    │ ✅ Auto-rebuild on push      │
│    hf.space                  │                    │                              │
│                              │                    │                              │
│ ✅ Auto-rebuild on push      │                    └──────────────────────────────┘
│                              │
└──────────────────────────────┘
```

---

## 🔐 Environment Variables Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENVIRONMENT VARIABLES SETUP                            │
└─────────────────────────────────────────────────────────────────────────────┘

LOCAL DEVELOPMENT
├── app/backend/.env
│   └── TMDB_API_KEY="your_key"
│
└── app/frontend/.env
    └── VITE_API_BASE_URL="http://localhost:8000"

                            ↓ (git push)

PRODUCTION - HUGGING FACE
├── Repository Secrets
│   └── TMDB_API_KEY="your_key"
│       (Used by Docker container)
│
└── app/backend/.env
    └── TMDB_API_KEY="your_key"
        (Copied into container)

PRODUCTION - VERCEL
├── Environment Variables
│   └── VITE_API_BASE_URL="https://mahmdshafee-emotion-detection-api.hf.space"
│       (Used during build)
│
└── app/frontend/.env
    └── VITE_API_BASE_URL="https://mahmdshafee-emotion-detection-api.hf.space"
        (Committed to repo)
```

---

## 📊 File Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WHERE EACH FILE GOES                                 │
└─────────────────────────────────────────────────────────────────────────────┘

HUGGING FACE SPACES (Backend)
├── Dockerfile                          ← Root of repository
├── app/
│   └── backend/
│       ├── main.py                     ← FastAPI application
│       ├── requirements.txt            ← Python dependencies
│       ├── startup.sh                  ← Startup script
│       └── .env                        ← TMDB_API_KEY
└── models/
    ├── classifier.pt                   ← Model weights (710MB)
    ├── config.json                     ← Model configuration
    ├── model.safetensors               ← Safetensors format (710MB)
    └── metrics.json                    ← Model metrics

VERCEL (Frontend)
├── app/frontend/
│   ├── src/
│   │   ├── App.jsx                     ← Main React component
│   │   ├── main.jsx                    ← Entry point
│   │   ├── index.css                   ← Global styles
│   │   └── components/
│   │       ├── MovieCard.jsx           ← Movie card component
│   │       └── MovieCarousel.jsx       ← Movie carousel component
│   ├── public/
│   │   └── vite.svg                    ← Static assets
│   ├── package.json                    ← Dependencies
│   ├── vite.config.js                  ← Vite configuration
│   ├── tailwind.config.js              ← Tailwind CSS config
│   ├── postcss.config.js               ← PostCSS config
│   ├── index.html                      ← HTML entry point
│   └── .env                            ← VITE_API_BASE_URL

NOT DEPLOYED (Local only)
├── notebooks/                          ← Jupyter notebooks
├── DeBERTa Test Results/               ← Test results
├── data/                               ← Training data
├── .git/                               ← Git history
└── README.md                           ← Documentation
```

---

## 🔄 Deployment Process Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT TIMELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

MINUTE 0-1: PREPARE
├── Update .env files
├── Commit changes
└── Ready to push

MINUTE 1-2: PUSH TO GITHUB
├── git push origin main
└── Code on GitHub

MINUTE 2-3: CREATE HUGGING FACE SPACE
├── Create new Space
├── Select Docker SDK
└── Space created

MINUTE 3-4: PUSH TO HUGGING FACE
├── git push huggingface main
└── Code on Hugging Face

MINUTE 4-14: HUGGING FACE BUILD
├── 0-2 min: Docker image build
├── 2-5 min: Install dependencies
├── 5-10 min: Load model
├── 10-12 min: Start application
└── 12-14 min: Ready (green status)

MINUTE 14-15: CREATE VERCEL PROJECT
├── Import GitHub repository
├── Configure settings
└── Project created

MINUTE 15-20: VERCEL BUILD & DEPLOY
├── 0-2 min: Install dependencies
├── 2-4 min: Build frontend
├── 4-5 min: Deploy to CDN
└── 5-20 min: Ready (green status)

MINUTE 20-25: TESTING
├── Test backend health
├── Test frontend load
├── Test emotion detection
└── Test movie display

MINUTE 25-30: VERIFICATION
├── Check logs
├── Verify CORS
├── Test integration
└── Ready for production!

TOTAL TIME: 30 MINUTES ✅
```

---

## 🎯 Deployment Checklist with Visuals

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT CHECKLIST                                   │
└─────────────────────────────────────────────────────────────────────────────┘

PREPARATION
  ☐ Get TMDB API key
  ☐ Create GitHub account
  ☐ Create Hugging Face account
  ☐ Create Vercel account

BACKEND SETUP
  ☐ Update app/backend/.env with TMDB_API_KEY
  ☐ Verify Dockerfile exists
  ☐ Verify models/ directory has all files
  ☐ Verify app/backend/ has all files

GITHUB
  ☐ Initialize git repository
  ☐ Add all files
  ☐ Create initial commit
  ☐ Push to GitHub

HUGGING FACE
  ☐ Create new Space
  ☐ Select Docker SDK
  ☐ Add TMDB_API_KEY to secrets
  ☐ Push code to Hugging Face
  ☐ Wait for build (5-10 minutes)
  ☐ Verify green status
  ☐ Test health endpoint

FRONTEND SETUP
  ☐ Update app/frontend/.env with backend URL
  ☐ Verify app/frontend/ has all files
  ☐ Commit changes
  ☐ Push to GitHub

VERCEL
  ☐ Create new project
  ☐ Import GitHub repository
  ☐ Set root directory to app/frontend
  ☐ Add VITE_API_BASE_URL environment variable
  ☐ Deploy
  ☐ Wait for build (2-5 minutes)
  ☐ Verify green status

TESTING
  ☐ Test backend health endpoint
  ☐ Test frontend loads
  ☐ Test emotion detection
  ☐ Test movie display
  ☐ Check browser console for errors
  ☐ Test on mobile

FINAL VERIFICATION
  ☐ Backend URL works
  ☐ Frontend URL works
  ☐ CORS configured correctly
  ☐ TMDB API key valid
  ☐ All features working
  ☐ No console errors

READY FOR PRODUCTION ✅
```

---

## 🚀 Quick Reference: What Goes Where

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE: DEPLOYMENT TARGETS                      │
└─────────────────────────────────────────────────────────────────────────────┘

HUGGING FACE SPACES (Backend)
┌─────────────────────────────────────────────────────────────────────────────┐
│ What: FastAPI backend with DeBERTa model                                    │
│ Where: https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api   │
│ URL: https://mahmdshafee-emotion-detection-api.hf.space                    │
│                                                                              │
│ Files to include:                                                            │
│ ✅ Dockerfile (root)                                                         │
│ ✅ app/backend/main.py                                                       │
│ ✅ app/backend/requirements.txt                                              │
│ ✅ app/backend/startup.sh                                                    │
│ ✅ app/backend/.env (with TMDB_API_KEY)                                      │
│ ✅ models/ (all model files)                                                 │
│                                                                              │
│ Files to exclude:                                                            │
│ ❌ app/frontend/ (not needed)                                                │
│ ❌ notebooks/ (not needed)                                                   │
│ ❌ data/ (not needed)                                                        │
│                                                                              │
│ Environment Variables:                                                       │
│ • TMDB_API_KEY (set in secrets)                                             │
│                                                                              │
│ Endpoints:                                                                   │
│ • GET /health                                                                │
│ • POST /recommendations                                                      │
│ • POST /predict                                                              │
│ • GET /emotions                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

VERCEL (Frontend)
┌─────────────────────────────────────────────────────────────────────────────┐
│ What: React frontend with Vite                                              │
│ Where: https://vercel.com/dashboard                                         │
│ URL: https://moodflix.vercel.app                                            │
│                                                                              │
│ Files to include:                                                            │
│ ✅ app/frontend/ (entire directory)                                          │
│ ✅ app/frontend/.env (with VITE_API_BASE_URL)                                │
│                                                                              │
│ Files to exclude:                                                            │
│ ❌ app/backend/ (not needed)                                                 │
│ ❌ models/ (not needed)                                                      │
│ ❌ Dockerfile (not needed)                                                   │
│                                                                              │
│ Configuration:                                                               │
│ • Root Directory: app/frontend                                              │
│ • Build Command: npm run build                                              │
│ • Output Directory: dist                                                    │
│                                                                              │
│ Environment Variables:                                                       │
│ • VITE_API_BASE_URL (set in Vercel dashboard)                               │
│                                                                              │
│ Features:                                                                    │
│ • Emotion input form                                                         │
│ • Movie recommendations display                                             │
│ • Dark/Light theme toggle                                                   │
│ • Responsive design                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Performance Expectations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PERFORMANCE EXPECTATIONS                               │
└─────────────────────────────────────────────────────────────────────────────┘

BACKEND (Hugging Face)
├── Startup Time
│   ├── Container start: 5-10 seconds
│   ├── Model loading: 30-60 seconds
│   └── Total: 1-2 minutes
│
├── Request Processing
│   ├── Emotion detection: 100-200ms
│   ├── Movie fetching: 500-1000ms
│   └── Total response: 1-2 seconds
│
└── Resource Usage
    ├── Memory: 1.5-2GB
    ├── CPU: 50-100% during inference
    └── Disk: 1.5GB (models)

FRONTEND (Vercel)
├── Page Load
│   ├── Initial load: <2 seconds
│   ├── Time to interactive: <3 seconds
│   └── Fully loaded: <5 seconds
│
├── User Interaction
│   ├── Submit button click: Instant
│   ├── Emotion detection: 1-2 seconds
│   ├── Movie display: Instant
│   └── Total user experience: 2-3 seconds
│
└── Resource Usage
    ├── Bundle size: ~500KB (gzipped)
    ├── Memory: 50-100MB
    └── Network: ~1-2MB per session

NETWORK
├── Frontend → Backend
│   ├── Request size: ~100 bytes
│   ├── Response size: ~50-100KB
│   └── Latency: 100-500ms
│
└── Backend → TMDB
    ├── Request size: ~200 bytes
    ├── Response size: ~50-100KB
    └── Latency: 200-500ms
```

---

**This visual guide should help you understand the complete deployment process!**

For step-by-step instructions, refer to: `COMPLETE_DEPLOYMENT_GUIDE.md`
