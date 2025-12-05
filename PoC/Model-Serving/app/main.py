from fastapi import FastAPI
from predictor import Predictor

app = FastAPI(title="HIMAS Mortality Prediction API")

# Load model + preprocessing artifacts ONCE at startup
predictor = Predictor()

@app.get("/")
def root():
    return {"status": "HIMAS prediction service running"}

@app.post("/predict")
def predict(payload: dict):
    """
    Payload should contain feature values in a flat JSON format:
    {
        "age": 45,
        "heart_rate": 120,
        "gender": "Female",
        ...
    }
    """
    try:
        prediction = predictor.predict(payload)
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}
