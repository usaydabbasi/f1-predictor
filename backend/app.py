from fastapi import FastAPI
import joblib, json
import numpy as np
from pathlib import Path

app = FastAPI(title="F1 Predictor API")

# Load model and features at startup
MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"
FEATURES_PATH = Path(__file__).parent / "models" / "features.json"

model = joblib.load(MODEL_PATH)
feature_order = json.load(open(FEATURES_PATH))

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict")
def predict(data: dict):
    try:
        x = np.array([[data[f] for f in feature_order]])
        prob = model.predict_proba(x)[0, 1]
        return {"win_probability": round(float(prob), 3)}
    except Exception as e:
        return {"error": str(e)}


