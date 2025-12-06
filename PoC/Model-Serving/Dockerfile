FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from app directory
COPY app/requirements.txt .

# Install TensorFlow CPU - match training version range
RUN pip install --no-cache-dir \
    "tensorflow-cpu>=2.11.1,<2.18.0" \
    && pip install --no-cache-dir -r requirements.txt

# Critical: Set TensorFlow to avoid CPU optimizations
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Copy application code from app directory
COPY app/*.py ./

# Copy model artifacts
COPY local_model/ /app/local_model/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
