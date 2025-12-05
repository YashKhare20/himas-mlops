"""
FastAPI Prediction Service (Skeleton)

This file defines the structure of the API that Vertex AI will call.
Later, we will connect:
- model loading
- preprocessor loading
- real inference logic
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import Dict, Any

# Placeholder: will replace with real loader
MODEL = None
PREPROCESSOR = None

app = FastAPI(title="HIMAS ICU Mortality Prediction Service")


# ---------- Request Schema ----------
class PatientInput(BaseModel):
    data: Dict[str, Any]   # One row of patient features


# ---------- Health Check ----------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------- Prediction Endpoint ----------
@app.post("/predict")
def predict(request: PatientInput):

    # 1. Convert dict → DataFrame
    df = pd.DataFrame([request.data])

    # Placeholder: Preprocessor not loaded yet
    if PREPROCESSOR is None:
        return {"error": "Preprocessor not loaded"}

    # Placeholder: Model not loaded yet
    if MODEL is None:
        return {"error": "Model not loaded"}

    # 2. Transform input
    X = PREPROCESSOR.transform(df)

    # 3. Predict
    y_proba = MODEL.predict(X)[0][0]
    y_pred = int(y_proba >= 0.5)

    return {
        "probability": float(y_proba),
        "prediction": y_pred
    }
