FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/torch_cache \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories
RUN mkdir -p /app/torch_cache /app/hf_cache /app/models

# Copy requirements first (for better layer caching)
COPY app/backend/requirements.txt .

# Install Python dependencies with specific versions for stability
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy models directory
COPY models/ /app/models/

# Copy application code
COPY app/backend/ /app/

# Copy and make startup script executable
COPY app/backend/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Run the application with startup script
CMD ["/app/startup.sh"]