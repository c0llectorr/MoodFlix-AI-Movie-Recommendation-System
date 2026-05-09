# MoodFlix Quick Deployment Reference Card

## 🚀 30-Minute Deployment Guide

---

## PART 1: BACKEND ON HUGGING FACE (10 minutes)

### Step 1: Push to GitHub
```bash
cd "c:\Users\mahma\Desktop\MoodFlix-AI-Movie-Recommendation-System"
git add .
git commit -m "Deploy MoodFlix backend"
git push origin main
```

### Step 2: Create Hugging Face Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. **Space name:** `emotion-detection-api`
4. **SDK:** Docker
5. **Visibility:** Public
6. Click "Create Space"

### Step 3: Push to Hugging Face
```bash
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api
git push huggingface main
```

### Step 4: Add TMDB API Key
1. Go to your Hugging Face Space
2. Click "Settings"
3. Scroll to "Repository secrets"
4. Click "Add a secret"
5. **Name:** `TMDB_API_KEY`
6. **Value:** Your TMDB API key
7. Click "Add secret"

### Step 5: Wait for Deployment
- ⏱️ 5-10 minutes for build
- 🟡 Yellow = Building
- 🟢 Green = Ready
- Check logs if red

### Step 6: Get Backend URL
Copy your Space URL: `https://mahmdshafee-emotion-detection-api.hf.space`

---

## PART 2: FRONTEND ON VERCEL (10 minutes)

### Step 1: Update Frontend .env
**File:** `app/frontend/.env`
```
VITE_API_BASE_URL="https://mahmdshafee-emotion-detection-api.hf.space"
```
Replace with your actual Hugging Face Space URL.

### Step 2: Push to GitHub
```bash
git add app/frontend/.env
git commit -m "Update backend URL"
git push origin main
```

### Step 3: Create Vercel Project
1. Go to https://vercel.com
2. Click "Add New..." → "Project"
3. Click "Import Git Repository"
4. Paste: `https://github.com/YOUR_USERNAME/MoodFlix-Backend`
5. Click "Import"

### Step 4: Configure Vercel
1. **Framework:** Vite
2. **Root Directory:** `app/frontend`
3. **Environment Variables:**
   - **Name:** `VITE_API_BASE_URL`
   - **Value:** Your Hugging Face Space URL
4. Click "Deploy"

### Step 5: Wait for Deployment
- ⏱️ 2-5 minutes
- 🟢 Green = Ready

### Step 6: Get Frontend URL
Your frontend URL: `https://moodflix.vercel.app` (or similar)

---

## PART 3: TEST (5 minutes)

### Test Backend
```bash
curl https://mahmdshafee-emotion-detection-api.hf.space/health
```
Should return: `{"status":"healthy","device":"cpu","model_loaded":true}`

### Test Frontend
1. Go to your Vercel URL
2. Type: "I am feeling amazing!"
3. Click "Detect & Suggest"
4. Should see emotion and movies

### Test Integration
1. Open browser DevTools (F12)
2. Go to "Network" tab
3. Submit text
4. Look for POST to `/recommendations`
5. Should see 200 status and movie data

---

## 📁 FILES DEPLOYED

### On Hugging Face (Backend):
```
Dockerfile                          ← Root
app/backend/
  ├── main.py                       ← FastAPI app
  ├── requirements.txt              ← Dependencies
  ├── startup.sh                    ← Startup script
  └── .env                          ← TMDB_API_KEY
models/
  ├── classifier.pt                 ← Model weights
  ├── config.json                   ← Config
  ├── model.safetensors             ← Safetensors
  └── metrics.json                  ← Metrics
```

### On Vercel (Frontend):
```
app/frontend/
  ├── src/
  │   ├── App.jsx                   ← Main component
  │   ├── main.jsx
  │   └── components/
  ├── public/
  ├── package.json
  ├── vite.config.js
  ├── tailwind.config.js
  ├── index.html
  └── .env                          ← VITE_API_BASE_URL
```

---

## 🔑 ENVIRONMENT VARIABLES

### Hugging Face (Backend)
**Secret Name:** `TMDB_API_KEY`
**Value:** Your TMDB API key from https://www.themoviedb.org/settings/api

### Vercel (Frontend)
**Name:** `VITE_API_BASE_URL`
**Value:** Your Hugging Face Space URL

---

## 🧪 QUICK TESTS

### Test 1: Health Check
```bash
curl https://mahmdshafee-emotion-detection-api.hf.space/health
```

### Test 2: Emotion Detection
```bash
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am happy"}'
```

### Test 3: Get Movies
```bash
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling amazing!"}'
```

---

## ⚠️ COMMON ISSUES & FIXES

| Issue | Cause | Fix |
|-------|-------|-----|
| 503 Service Unavailable | Model loading | Wait 2-3 minutes |
| "Failed to fetch" | Backend URL wrong | Update .env in frontend |
| CORS error | Frontend URL not in CORS | Add to main.py allow_origins |
| No movies | TMDB_API_KEY invalid | Verify key at TMDB website |
| Blank page | Frontend not deployed | Check Vercel logs |
| "Model not found" | Models not copied | Rebuild Hugging Face Space |

---

## 📊 EXPECTED TIMES

| Task | Time |
|------|------|
| Push to GitHub | 1 min |
| Create HF Space | 2 min |
| Build Docker image | 5-10 min |
| Model loading | 1-2 min |
| **Backend Total** | **10-15 min** |
| Update frontend .env | 1 min |
| Create Vercel project | 2 min |
| Build & deploy | 2-5 min |
| **Frontend Total** | **5-10 min** |
| **TOTAL** | **20-30 min** |

---

## 🔗 IMPORTANT URLS

| Service | URL |
|---------|-----|
| GitHub Repo | https://github.com/YOUR_USERNAME/MoodFlix-Backend |
| HF Space | https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api |
| Backend API | https://mahmdshafee-emotion-detection-api.hf.space |
| Frontend | https://moodflix.vercel.app |
| API Docs | https://mahmdshafee-emotion-detection-api.hf.space/docs |

---

## ✅ DEPLOYMENT CHECKLIST

### Backend
- [ ] GitHub repo created
- [ ] Code pushed to GitHub
- [ ] Hugging Face Space created
- [ ] Code pushed to Hugging Face
- [ ] TMDB_API_KEY added to secrets
- [ ] Deployment complete (green status)
- [ ] Health endpoint works

### Frontend
- [ ] .env updated with backend URL
- [ ] Code pushed to GitHub
- [ ] Vercel project created
- [ ] Root directory set to app/frontend
- [ ] VITE_API_BASE_URL environment variable set
- [ ] Deployment complete
- [ ] Frontend loads

### Integration
- [ ] Backend and frontend can communicate
- [ ] No CORS errors
- [ ] Emotion detection works
- [ ] Movies display

---

## 🎯 NEXT STEPS

1. ✅ Deploy backend to Hugging Face
2. ✅ Deploy frontend to Vercel
3. ✅ Test both services
4. ✅ Share your app!

---

## 💡 TIPS

- **Bookmark your URLs** for easy access
- **Save your TMDB API key** somewhere safe
- **Monitor logs** during first deployment
- **Test on mobile** to verify responsive design
- **Share your app** with friends!

---

**Ready to deploy? Let's go! 🚀**

For detailed instructions, see: `COMPLETE_DEPLOYMENT_GUIDE.md`
