# 🚀 MoodFlix Deployment Documentation

Welcome! This folder contains everything you need to deploy MoodFlix to production.

## 📚 Documentation Files

### 1. **COMPLETE_DEPLOYMENT_GUIDE.md** ⭐ START HERE
   - **Length:** Comprehensive (detailed)
   - **Best for:** Step-by-step instructions
   - **Contains:**
     - Prerequisites and setup
     - Part 1: Backend deployment on Hugging Face
     - Part 2: Frontend deployment on Vercel
     - Part 3: Connecting frontend to backend
     - Verification and testing
     - Troubleshooting guide
   - **Time to read:** 30 minutes
   - **Time to deploy:** 20-30 minutes

### 2. **QUICK_DEPLOYMENT_REFERENCE.md** ⚡ QUICK START
   - **Length:** Concise (quick reference)
   - **Best for:** Experienced developers
   - **Contains:**
     - 30-minute deployment summary
     - Command-by-command instructions
     - File checklist
     - Common issues and fixes
   - **Time to read:** 5 minutes
   - **Time to deploy:** 20-30 minutes

### 3. **DEPLOYMENT_VISUAL_GUIDE.md** 📊 VISUAL LEARNER
   - **Length:** Medium (with diagrams)
   - **Best for:** Understanding architecture
   - **Contains:**
     - Architecture diagrams
     - Data flow visualization
     - Deployment process timeline
     - File distribution map
     - Performance expectations
   - **Time to read:** 15 minutes

### 4. **DEPLOYMENT_CHECKLIST.md** ✅ VERIFICATION
   - **Length:** Medium (checklist format)
   - **Best for:** Tracking progress
   - **Contains:**
     - Issue fixes summary
     - Files modified/created
     - Deployment steps
     - Verification checklist
     - Troubleshooting guide
   - **Time to read:** 10 minutes

---

## 🎯 Quick Start (5 Minutes)

### Prerequisites
- ✅ GitHub account
- ✅ Hugging Face account
- ✅ Vercel account
- ✅ TMDB API key (get from https://www.themoviedb.org/settings/api)

### Backend (10 minutes)
```bash
# 1. Push to GitHub
git push origin main

# 2. Create Hugging Face Space
# Go to https://huggingface.co/spaces → Create new Space
# Name: emotion-detection-api, SDK: Docker

# 3. Push to Hugging Face
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api
git push huggingface main

# 4. Add TMDB_API_KEY to Hugging Face secrets
# Go to Space Settings → Repository secrets → Add TMDB_API_KEY

# 5. Wait for deployment (5-10 minutes)
```

### Frontend (10 minutes)
```bash
# 1. Update frontend .env
# Edit: app/frontend/.env
# Set: VITE_API_BASE_URL="https://mahmdshafee-emotion-detection-api.hf.space"

# 2. Push to GitHub
git push origin main

# 3. Create Vercel project
# Go to https://vercel.com → Add New → Project
# Import your GitHub repository

# 4. Configure Vercel
# Root Directory: app/frontend
# Environment Variable: VITE_API_BASE_URL=your_backend_url

# 5. Deploy (2-5 minutes)
```

### Test (5 minutes)
```bash
# Test backend
curl https://mahmdshafee-emotion-detection-api.hf.space/health

# Test frontend
# Go to https://moodflix.vercel.app
# Type text and submit
```

---

## 📋 Which Guide Should I Read?

### I'm new to deployment
→ Read **COMPLETE_DEPLOYMENT_GUIDE.md**

### I've deployed before and want quick instructions
→ Read **QUICK_DEPLOYMENT_REFERENCE.md**

### I want to understand the architecture
→ Read **DEPLOYMENT_VISUAL_GUIDE.md**

### I want to verify everything is correct
→ Read **DEPLOYMENT_CHECKLIST.md**

### I'm having issues
→ Check **Troubleshooting** section in any guide

---

## 🔑 Key Information

### Backend (Hugging Face Spaces)
- **URL:** `https://mahmdshafee-emotion-detection-api.hf.space`
- **Files:** Dockerfile, app/backend/, models/
- **Environment:** TMDB_API_KEY
- **Port:** 7860

### Frontend (Vercel)
- **URL:** `https://moodflix.vercel.app`
- **Files:** app/frontend/
- **Environment:** VITE_API_BASE_URL
- **Framework:** React + Vite

### External Services
- **TMDB API:** https://www.themoviedb.org/settings/api
- **GitHub:** https://github.com
- **Hugging Face:** https://huggingface.co/spaces
- **Vercel:** https://vercel.com

---

## 📁 Files Included in This Deployment

### Backend Files (Deploy to Hugging Face)
```
Dockerfile                          ← Docker configuration
app/backend/
  ├── main.py                       ← FastAPI application
  ├── requirements.txt              ← Python dependencies
  ├── startup.sh                    ← Startup script
  └── .env                          ← Environment variables
models/
  ├── classifier.pt                 ← Model weights
  ├── config.json                   ← Model config
  ├── model.safetensors             ← Safetensors format
  └── metrics.json                  ← Model metrics
```

### Frontend Files (Deploy to Vercel)
```
app/frontend/
  ├── src/
  │   ├── App.jsx                   ← Main component
  │   ├── main.jsx                  ← Entry point
  │   └── components/               ← React components
  ├── public/                        ← Static assets
  ├── package.json                  ← Dependencies
  ├── vite.config.js                ← Vite config
  ├── tailwind.config.js            ← Tailwind config
  ├── index.html                    ← HTML entry
  └── .env                          ← Environment variables
```

---

## ⏱️ Deployment Timeline

| Step | Time | Status |
|------|------|--------|
| Prepare files | 2 min | ✅ |
| Push to GitHub | 1 min | ✅ |
| Create HF Space | 2 min | ✅ |
| Build backend | 10 min | ⏳ |
| Create Vercel project | 2 min | ✅ |
| Build frontend | 5 min | ⏳ |
| Test integration | 5 min | ✅ |
| **TOTAL** | **~30 min** | ✅ |

---

## 🧪 Testing Checklist

### Backend Tests
- [ ] Health endpoint returns 200
- [ ] Emotion detection works
- [ ] Movies are fetched
- [ ] CORS headers present

### Frontend Tests
- [ ] Page loads without errors
- [ ] Can type text
- [ ] Can submit form
- [ ] Receives emotion result
- [ ] Displays movies

### Integration Tests
- [ ] Frontend connects to backend
- [ ] No CORS errors
- [ ] Full flow works end-to-end
- [ ] Works on mobile

---

## 🆘 Need Help?

### Common Issues

**503 Service Unavailable**
- Wait 2-3 minutes for model to load
- Check Hugging Face logs

**"Failed to fetch"**
- Verify backend URL in frontend .env
- Check backend is running

**CORS Error**
- Add frontend URL to CORS in main.py
- Rebuild backend

**No movies displaying**
- Verify TMDB_API_KEY is valid
- Check backend logs

### Get More Help
1. Check the **Troubleshooting** section in the deployment guide
2. Review the **DEPLOYMENT_CHECKLIST.md**
3. Check Hugging Face/Vercel logs
4. Test endpoints with curl

---

## 📞 Support Resources

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Hugging Face Spaces:** https://huggingface.co/docs/hub/spaces
- **Vercel Docs:** https://vercel.com/docs
- **TMDB API:** https://developer.themoviedb.org/docs
- **React Docs:** https://react.dev/
- **Vite Docs:** https://vitejs.dev/

---

## ✨ What's New in This Version

### Fixed Issues
- ✅ 503 Service Unavailable errors
- ✅ "Your space is not valid JSON" errors
- ✅ Network connectivity issues
- ✅ Model path problems
- ✅ CORS configuration

### Improvements
- ✅ Better error handling
- ✅ Graceful startup
- ✅ Improved logging
- ✅ Model readiness checks
- ✅ Proper JSON responses

### New Features
- ✅ Startup script with debugging
- ✅ Health check endpoint
- ✅ Model readiness middleware
- ✅ Better error messages
- ✅ TMDB movie integration

---

## 🎯 Next Steps

1. **Read the appropriate guide** based on your experience level
2. **Gather prerequisites** (API keys, accounts)
3. **Follow the step-by-step instructions**
4. **Test your deployment**
5. **Share your app!**

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR MOODFLIX APP                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frontend (Vercel)          Backend (Hugging Face)          │
│  ┌──────────────────┐       ┌──────────────────────┐        │
│  │ React + Vite     │◄─────►│ FastAPI + DeBERTa    │        │
│  │ Emotion Input    │ HTTPS │ Model Loading        │        │
│  │ Movie Display    │       │ TMDB Integration     │        │
│  └──────────────────┘       └──────────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Ready to Deploy?

**Start with:** `COMPLETE_DEPLOYMENT_GUIDE.md`

**Quick version:** `QUICK_DEPLOYMENT_REFERENCE.md`

**Visual guide:** `DEPLOYMENT_VISUAL_GUIDE.md`

---

**Good luck with your deployment! 🎉**

For detailed instructions, see the deployment guides above.
