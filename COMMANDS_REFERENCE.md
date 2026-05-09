# MoodFlix Deployment Commands Reference

Quick copy-paste commands for deployment.

---

## 🔧 Git Commands

### Initialize Git (if not already done)
```bash
cd "c:\Users\mahma\Desktop\MoodFlix-AI-Movie-Recommendation-System"
git init
git add .
git commit -m "Initial commit: MoodFlix with fixed deployment"
git remote add origin https://github.com/YOUR_USERNAME/MoodFlix-Backend.git
git branch -M main
git push -u origin main
```

### Push to GitHub
```bash
git add .
git commit -m "Update: [describe changes]"
git push origin main
```

### Add Hugging Face Remote
```bash
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api
```

### Push to Hugging Face
```bash
git push huggingface main
```

### Push to Both GitHub and Hugging Face
```bash
git push origin main
git push huggingface main
```

### Check Git Status
```bash
git status
```

### View Git Log
```bash
git log --oneline
```

---

## 🐳 Docker Commands (Local Testing)

### Build Docker Image
```bash
docker build -t moodflix-api .
```

### Run Docker Container
```bash
docker run -p 7860:7860 \
  -e TMDB_API_KEY="your_api_key" \
  moodflix-api
```

### Run with Volume Mount (for development)
```bash
docker run -p 7860:7860 \
  -e TMDB_API_KEY="your_api_key" \
  -v "c:\Users\mahma\Desktop\MoodFlix-AI-Movie-Recommendation-System\app\backend:/app" \
  moodflix-api
```

### View Container Logs
```bash
docker logs <container_id>
```

### List Running Containers
```bash
docker ps
```

### Stop Container
```bash
docker stop <container_id>
```

### Remove Image
```bash
docker rmi moodflix-api
```

---

## 🌐 API Testing Commands

### Test Health Endpoint
```bash
curl https://mahmdshafee-emotion-detection-api.hf.space/health
```

### Test Emotions Endpoint
```bash
curl https://mahmdshafee-emotion-detection-api.hf.space/emotions
```

### Test Emotion Detection
```bash
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'
```

### Test Movie Recommendations
```bash
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling amazing!"}'
```

### Test with Different Emotions
```bash
# Joy
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy and excited!"}'

# Sadness
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling sad and lonely"}'

# Anger
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so angry and frustrated!"}'

# Fear
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I am scared and worried"}'

# Love
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "I love you so much!"}'

# Surprise
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "Wow! I cannot believe this!"}'

# Neutral
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/recommendations \
  -H "Content-Type: application/json" \
  -d '{"text": "This is just a regular day"}'
```

### Test Batch Prediction
```bash
curl -X POST https://mahmdshafee-emotion-detection-api.hf.space/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I am happy",
      "I am sad",
      "I am angry"
    ]
  }'
```

### Pretty Print JSON Response
```bash
# Using jq (if installed)
curl -s https://mahmdshafee-emotion-detection-api.hf.space/health | jq .

# Or using Python
curl -s https://mahmdshafee-emotion-detection-api.hf.space/health | python -m json.tool
```

---

## 📝 File Editing Commands

### Update Frontend .env
```bash
# Windows PowerShell
Set-Content -Path "app/frontend/.env" -Value 'VITE_API_BASE_URL="https://mahmdshafee-emotion-detection-api.hf.space"'

# Or manually edit the file
# File: app/frontend/.env
# Content: VITE_API_BASE_URL="https://mahmdshafee-emotion-detection-api.hf.space"
```

### Update Backend .env
```bash
# Windows PowerShell
Set-Content -Path "app/backend/.env" -Value 'TMDB_API_KEY="your_actual_api_key"'

# Or manually edit the file
# File: app/backend/.env
# Content: TMDB_API_KEY="your_actual_api_key"
```

### View File Contents
```bash
# Windows PowerShell
Get-Content "app/frontend/.env"
Get-Content "app/backend/.env"

# Or use cat
cat app/frontend/.env
cat app/backend/.env
```

---

## 📦 Node.js Commands (Frontend)

### Install Dependencies
```bash
cd app/frontend
npm install
```

### Run Development Server
```bash
npm run dev
```

### Build for Production
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

### Lint Code
```bash
npm run lint
```

---

## 🐍 Python Commands (Backend)

### Create Virtual Environment
```bash
python -m venv venv
```

### Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r app/backend/requirements.txt
```

### Run Backend Locally
```bash
cd app/backend
python main.py
```

### Run with Uvicorn
```bash
cd app/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Check Python Version
```bash
python --version
```

### Compile Python File (Check Syntax)
```bash
python -m py_compile app/backend/main.py
```

---

## 🔍 Debugging Commands

### Check if Port is in Use
```bash
# Windows
netstat -ano | findstr :7860
netstat -ano | findstr :8000
netstat -ano | findstr :3000

# macOS/Linux
lsof -i :7860
lsof -i :8000
lsof -i :3000
```

### Kill Process on Port
```bash
# Windows
taskkill /PID <PID> /F

# macOS/Linux
kill -9 <PID>
```

### Check Network Connectivity
```bash
ping mahmdshafee-emotion-detection-api.hf.space
ping api.themoviedb.org
```

### Test DNS Resolution
```bash
nslookup mahmdshafee-emotion-detection-api.hf.space
```

---

## 📊 Monitoring Commands

### Check Disk Space
```bash
# Windows
Get-Volume

# macOS/Linux
df -h
```

### Check Memory Usage
```bash
# Windows
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10

# macOS/Linux
top
```

### Check CPU Usage
```bash
# Windows
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10

# macOS/Linux
ps aux --sort=-%cpu | head -10
```

---

## 🔐 Security Commands

### Generate Random API Key (for testing)
```bash
# Windows PowerShell
[System.Guid]::NewGuid().ToString()

# macOS/Linux
openssl rand -hex 16
```

### Check if File Contains Secrets
```bash
# Search for common secret patterns
grep -r "api_key\|password\|secret" app/backend/

# Search for TMDB key
grep -r "93bbcf4a92e0749987e9607fb28663a6" .
```

---

## 📋 Verification Commands

### Verify All Files Exist
```bash
# Windows PowerShell
Test-Path "Dockerfile"
Test-Path "app/backend/main.py"
Test-Path "app/backend/requirements.txt"
Test-Path "app/backend/startup.sh"
Test-Path "app/backend/.env"
Test-Path "models/classifier.pt"
Test-Path "models/config.json"
Test-Path "models/model.safetensors"
Test-Path "app/frontend/package.json"
Test-Path "app/frontend/.env"
```

### List Directory Structure
```bash
# Windows PowerShell
Get-ChildItem -Recurse -Depth 2

# macOS/Linux
tree -L 2
```

### Count Files
```bash
# Windows PowerShell
(Get-ChildItem -Recurse).Count

# macOS/Linux
find . -type f | wc -l
```

---

## 🚀 One-Liner Deployment Commands

### Complete Backend Deployment
```bash
git add . && git commit -m "Deploy backend" && git push origin main && git push huggingface main
```

### Complete Frontend Deployment
```bash
git add app/frontend/.env && git commit -m "Update backend URL" && git push origin main
```

### Test Everything
```bash
curl https://mahmdshafee-emotion-detection-api.hf.space/health && echo "Backend OK" && curl https://moodflix.vercel.app && echo "Frontend OK"
```

---

## 📱 Browser Console Commands

### Test API from Browser Console
```javascript
// Test health endpoint
fetch('https://mahmdshafee-emotion-detection-api.hf.space/health')
  .then(r => r.json())
  .then(d => console.log(d))

// Test recommendations
fetch('https://mahmdshafee-emotion-detection-api.hf.space/recommendations', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'I am happy'})
})
  .then(r => r.json())
  .then(d => console.log(d))
```

---

## 🔗 Useful URLs

### Development
- Local Frontend: http://localhost:3000
- Local Backend: http://localhost:8000
- Local API Docs: http://localhost:8000/docs

### Production
- Frontend: https://moodflix.vercel.app
- Backend: https://mahmdshafee-emotion-detection-api.hf.space
- API Docs: https://mahmdshafee-emotion-detection-api.hf.space/docs

### Services
- GitHub: https://github.com
- Hugging Face: https://huggingface.co/spaces
- Vercel: https://vercel.com
- TMDB: https://www.themoviedb.org/settings/api

---

## 💾 Backup Commands

### Backup Project
```bash
# Windows PowerShell
Copy-Item -Path "MoodFlix-AI-Movie-Recommendation-System" -Destination "MoodFlix-AI-Movie-Recommendation-System-backup" -Recurse

# macOS/Linux
cp -r MoodFlix-AI-Movie-Recommendation-System MoodFlix-AI-Movie-Recommendation-System-backup
```

### Create Git Backup
```bash
git bundle create moodflix-backup.bundle --all
```

### Restore from Bundle
```bash
git clone moodflix-backup.bundle
```

---

## 🧹 Cleanup Commands

### Remove Docker Images
```bash
docker rmi moodflix-api
docker system prune -a
```

### Remove Node Modules
```bash
cd app/frontend
rm -r node_modules
npm install
```

### Remove Python Cache
```bash
# Windows PowerShell
Get-ChildItem -Path . -Include __pycache__ -Recurse | Remove-Item -Recurse

# macOS/Linux
find . -type d -name __pycache__ -exec rm -r {} +
```

### Clean Git
```bash
git gc
git prune
```

---

## 📞 Emergency Commands

### If Backend is Down
```bash
# Rebuild on Hugging Face
git push huggingface main

# Check logs
# Go to: https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-api
# Click: Logs tab
```

### If Frontend is Down
```bash
# Rebuild on Vercel
git push origin main

# Check logs
# Go to: https://vercel.com/dashboard
# Click: Deployments
```

### If You Need to Rollback
```bash
# View previous commits
git log --oneline

# Revert to previous commit
git revert <commit_hash>
git push origin main
```

---

## 🎯 Quick Command Sequences

### Full Deployment Sequence
```bash
# 1. Update files
# Edit app/frontend/.env with backend URL
# Edit app/backend/.env with TMDB_API_KEY

# 2. Commit and push
git add .
git commit -m "Deploy MoodFlix"
git push origin main
git push huggingface main

# 3. Wait for builds
# Monitor Hugging Face: 5-10 minutes
# Monitor Vercel: 2-5 minutes

# 4. Test
curl https://mahmdshafee-emotion-detection-api.hf.space/health
# Visit https://moodflix.vercel.app
```

### Local Testing Sequence
```bash
# 1. Install dependencies
cd app/frontend && npm install
cd ../backend && pip install -r requirements.txt

# 2. Start backend
cd app/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Start frontend (in new terminal)
cd app/frontend
npm run dev

# 4. Test
# Visit http://localhost:3000
# Type text and submit
```

---

**Save this file for quick reference during deployment!**
