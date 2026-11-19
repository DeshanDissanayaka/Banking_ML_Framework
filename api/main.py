from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse
import joblib, yaml, pandas as pd

app = FastAPI(title="Banking ML Prediction API", version="0.1.0")
cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(cfg["training"]["model_path"])
    except Exception:
        model = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        frame = pd.DataFrame(request.records)
        probs = model.predict_proba(frame)
        classes = list(model.classes_)
        positive = cfg["data"]["positive_label"]
        idx = classes.index(positive)
        preds = model.predict(frame)
        return {"predictions": [
            {"predicted_class": str(p), "probability_positive": float(pr[idx])}
            for p, pr in zip(preds, probs)
        ]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid prediction input: {exc}")
