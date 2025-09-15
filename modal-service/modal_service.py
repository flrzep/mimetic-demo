# Skeleton Modal service; replace with actual Modal functions if using modal framework
import base64
import io
import os
import time
from typing import List

from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Modal GPU Service (Mock)")

class PredictionResult(BaseModel):
    class_id: int
    confidence: float
    label: str | None = None

class PredictRequest(BaseModel):
    image: str  # base64

@app.get("/health")
async def health():
    return {"status": "ok", "model": "mock", "gpu": False}

@app.post("/predict")
async def predict(req: PredictRequest):
    # Mock decode and return dummy result
    _ = Image.open(io.BytesIO(base64.b64decode(req.image)))
    return {"success": True, "predictions": [{"class_id": 0, "confidence": 0.95, "label": "mock", "bbox": {"x": 50, "y": 50, "width": 100, "height": 100}}]}
