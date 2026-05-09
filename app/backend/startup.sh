#!/bin/bash
set -e

echo "===== Application Startup at $(date) ====="
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Model path: /app/models"
echo "Models available:"
ls -lah /app/models/ || echo "No models directory found"

echo ""
echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 7860 --log-level info
