#!/bin/bash
set -e

echo "📦 Downloading model artifacts from GCS..."

# Get the latest model timestamp
TIMESTAMP=$(gsutil cat gs://himas-mlops-models/model-registry/latest.txt)
echo "Using model from: $TIMESTAMP"

# Download model
mkdir -p /app/local_model/preprocessing
gsutil cp gs://himas-mlops-models/models/${TIMESTAMP}/model.keras /app/local_model/

# Download preprocessing artifacts
gsutil -m cp gs://himas-mlops-models/models/${TIMESTAMP}/preprocessing/*.pkl /app/local_model/preprocessing/

echo "✅ Model artifacts downloaded"
ls -lR /app/local_model/

# Start the application with python3
exec python3 -m uvicorn main:app --host 0.0.0.0 --port ${PORT}
