import os
import io
import time
import json
import base64
import asyncio
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from models import PredictionRequest, PredictionResponse, PredictionResult, HealthResponse
from utils.helpers import validate_image_file, convert_to_base64, resize_image_if_needed, log_request

APP_NAME = os.getenv("APP_NAME", "CV Backend")
MODAL_ENDPOINT_URL = os.getenv("MODAL_ENDPOINT_URL", "")
MODAL_API_KEY = os.getenv("MODAL_API_KEY", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

app = FastAPI(title=APP_NAME, default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class RateLimiter:
    def __init__(self, requests: int = 100, window: int = 3600):
        self.requests = requests
        self.window = window
        self.store = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        data = self.store.get(key, {"count": 0, "reset": now + self.window})
        if now > data["reset"]:
            data = {"count": 0, "reset": now + self.window}
        data["count"] += 1
        self.store[key] = data
        return data["count"] <= self.requests

limiter = RateLimiter(
    requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
    window=int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
)


@app.get("/")
async def root():
    return {"name": APP_NAME, "status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/health", response_model=HealthResponse)
async def health():
    start = time.perf_counter()
    modal_status = {"ok": False, "latency_ms": None}
    try:
        if MODAL_ENDPOINT_URL:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.get(f"{MODAL_ENDPOINT_URL}/health")
                modal_status["ok"] = r.status_code == 200
                modal_status["latency_ms"] = r.elapsed.total_seconds() * 1000 if r.elapsed else None
    except Exception:
        modal_status["ok"] = False
    backend_status = "ok"
    elapsed = time.perf_counter() - start
    return HealthResponse(
        status="ok",
        backend_status=backend_status,
        modal_status=modal_status,
        response_time=elapsed,
        timestamp=datetime.utcnow().isoformat()
    )


async def forward_to_modal(image_b64: str) -> list[PredictionResult]:
    if not MODAL_ENDPOINT_URL:
        # Mock prediction for local dev
        await asyncio.sleep(0.2)
        return [PredictionResult(class_id=0, confidence=0.95, label="mock")]
    headers = {"Authorization": f"Bearer {MODAL_API_KEY}"} if MODAL_API_KEY else {}
    payload = {"image": image_b64}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(f"{MODAL_ENDPOINT_URL}/predict", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        preds = data.get("predictions", [])
        return [PredictionResult(**p) for p in preds]


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    client_key = "default"
    if not limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    contents = await file.read()
    ok, msg = validate_image_file(file.filename, file.content_type or '', MAX_FILE_SIZE, len(contents))
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    t0 = time.perf_counter()
    try:
        resized = resize_image_if_needed(contents)
        image_b64 = convert_to_base64(resized)
        preds = await forward_to_modal(image_b64)
        elapsed = time.perf_counter() - t0
        resp = PredictionResponse(success=True, predictions=preds, processing_time=elapsed)
        log_request("/predict", elapsed, True)
        return resp
    except httpx.TimeoutException:
        elapsed = time.perf_counter() - t0
        log_request("/predict", elapsed, False)
        return PredictionResponse(success=False, error="Upstream timeout", processing_time=elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        log_request("/predict", elapsed, False)
        return PredictionResponse(success=False, error=str(e), processing_time=elapsed)


@app.post("/predict-base64", response_model=PredictionResponse)
async def predict_base64(req: PredictionRequest):
    t0 = time.perf_counter()
    try:
        preds = await forward_to_modal(req.image)
        elapsed = time.perf_counter() - t0
        return PredictionResponse(success=True, predictions=preds, processing_time=elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return PredictionResponse(success=False, error=str(e), processing_time=elapsed)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_json({"type": "welcome", "message": "Connected"})
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            data = await websocket.receive_text()
            # echo progress or commands
            await websocket.send_json({"type": "echo", "data": data})
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
