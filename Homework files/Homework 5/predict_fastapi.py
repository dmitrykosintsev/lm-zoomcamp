import pickle
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI(title="lead-score-prediction")

# Load model when server starts
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)
    return {
        "lead_score": prob
    }