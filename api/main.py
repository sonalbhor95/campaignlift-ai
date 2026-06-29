from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import MODEL_BUNDLE_FILE
from src.predict import load_model_bundle, predict_customer

app = FastAPI(title="CampaignLift AI API", version="1.0.0")
bundle = load_model_bundle(MODEL_BUNDLE_FILE) if MODEL_BUNDLE_FILE.exists() else None


class CustomerInput(BaseModel):
    f0: float = 0.0
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0
    f4: float = 0.0
    f5: float = 0.0
    f6: float = 0.0
    f7: float = 0.0
    f8: float = 0.0
    f9: float = 0.0
    f10: float = 0.0
    f11: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}


@app.post("/predict")
def predict(payload: CustomerInput):
    global bundle
    if bundle is None:
        bundle = load_model_bundle(MODEL_BUNDLE_FILE)
    return predict_customer(payload.model_dump(), bundle)
