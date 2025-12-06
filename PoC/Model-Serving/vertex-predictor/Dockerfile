FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the predictor code
COPY predictor.py .

# Set environment variables for Vertex AI
ENV AIP_STORAGE_URI=""
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080

# Expose port
EXPOSE 8080

# Run the Vertex AI model server
CMD ["python", "-m", "google.cloud.aiplatform.prediction.model_server", "--http_port=8080"]
