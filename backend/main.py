import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import tempfile
import time
import uuid
from datetime import datetime
from email.mime import image
from typing import AsyncGenerator, List, Optional, Tuple
from urllib.parse import urlparse

import aiofiles
import cv2
import httpx
import httpx as _httpx
import numpy as np
from cachetools import TTLCache
from fastapi import (Depends, FastAPI, File, HTTPException, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
from jose import jwt
from models import (BoundingBox, HealthResponse, PredictionResponse,
                    PredictionResult, VideoFrame, VideoProcessingRequest,
                    VideoProcessingResponse)
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketState
from utils.helpers import (convert_to_base64, log_processing_step, log_request,
                           log_video_processing, resize_image_if_needed,
                           validate_image_file, validate_video_file)

# Configure main logger
logger = logging.getLogger(__name__)

# Environment detection
IS_PRODUCTION = os.getenv("NODE_ENV") == "production" or os.getenv("RENDER") == "true"
DEBUG_MODE = not IS_PRODUCTION
USE_MOCK_MODAL = os.getenv("USE_MOCK_MODAL", "true" if not IS_PRODUCTION else "false").lower() == "true"

APP_NAME = os.getenv("APP_NAME", "CV Backend")
MODAL_ENDPOINT_URL = os.getenv("MODAL_ENDPOINT_URL", "")
MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID", "")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET", "")

# Debug: Log what we're getting from environment
logger.info(f"MODAL_ENDPOINT_URL from env: {MODAL_ENDPOINT_URL}")
logger.info(f"MODAL_TOKEN_ID configured: {bool(MODAL_TOKEN_ID)}")
logger.info(f"MODAL_TOKEN_SECRET configured: {bool(MODAL_TOKEN_SECRET)}")
if DEBUG_MODE:
    logger.info(f"Environment variables loaded: {list(os.environ.keys())}")

# Dynamic CORS based on environment
DEFAULT_CORS = "https://mimetic-demo*,http://localhost:3000" if IS_PRODUCTION else "http://localhost:3000"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", DEFAULT_CORS).split(",")

def is_cors_allowed(origin: str, allowed_patterns: List[str]) -> bool:
    """Check if an origin matches any of the allowed CORS patterns (supports wildcards)"""
    if not origin:
        return False
    
    logger.debug(f"Checking CORS for origin: {origin} against patterns: {allowed_patterns}")
    
    for pattern in allowed_patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
            
        # Exact match
        if origin == pattern:
            logger.debug(f"CORS allowed: exact match with {pattern}")
            return True
            
        # Wildcard pattern matching for Vercel URLs
        if '*' in pattern:
            # Convert wildcard pattern to regex
            regex_pattern = re.escape(pattern).replace(r'\*', '.*')
            if re.fullmatch(regex_pattern, origin):
                logger.debug(f"CORS allowed: wildcard match {pattern} -> {regex_pattern}")
                return True
    
    logger.warning(f"CORS blocked for origin: {origin}")
    return False

# Process CORS origins to support dynamic validation
CORS_PATTERNS = [o.strip() for o in CORS_ORIGINS if o.strip()]

class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        
        response = await call_next(request)
        
        # Handle CORS for all requests
        if origin and is_cors_allowed(origin, CORS_PATTERNS):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            if origin and is_cors_allowed(origin, CORS_PATTERNS):
                response = Response()
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "*"
                response.headers["Access-Control-Allow-Headers"] = "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Max-Age"] = "86400"
        
        return response

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID", "").strip()
SUPABASE_JWKS_URL = f"https://{SUPABASE_PROJECT_ID}.supabase.co/auth/v1/keys" if SUPABASE_PROJECT_ID else ""
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
_jwks_cache = TTLCache(maxsize=1, ttl=3600)

# Log environment info
logger.info(f"Environment: {'Production' if IS_PRODUCTION else 'Development'}")
logger.info(f"Debug mode: {DEBUG_MODE}")
logger.info(f"Use mock Modal: {USE_MOCK_MODAL}")
logger.info(f"CORS patterns: {CORS_PATTERNS}")
logger.info(f"Authentication required: {REQUIRE_AUTH}")

app = FastAPI(title=APP_NAME, default_response_class=ORJSONResponse)

# Add custom CORS middleware with wildcard support
app.add_middleware(CustomCORSMiddleware)

async def get_jwks():
  if not SUPABASE_JWKS_URL:
      return None
  jwks = _jwks_cache.get('jwks')
  if jwks:
      return jwks
  async with _httpx.AsyncClient(timeout=10) as client:
      r = await client.get(SUPABASE_JWKS_URL)
      r.raise_for_status()
      jwks = r.json()
      _jwks_cache['jwks'] = jwks
      return jwks

async def verify_bearer(token: Optional[str] = None):
    # If authentication is not required, allow anonymous access
    if not REQUIRE_AUTH:
        return None
    
    if not SUPABASE_JWKS_URL:
        return None  # allow anonymous if not configured
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization token")
    jwks = await get_jwks()
    if not jwks:
        raise HTTPException(status_code=500, detail="Unable to retrieve JWKS")
    try:
        unverified = jwt.get_unverified_header(token)
        kid = unverified.get('kid')
        key = next((k for k in jwks.get('keys', []) if k.get('kid') == kid), None)
        if not key:
            raise HTTPException(status_code=401, detail="Invalid token key")
        payload = jwt.decode(
            token,
            key,
            algorithms=[key.get('alg', 'RS256')],
            audience=f"authenticated",
            options={"verify_at_hash": False}
        )
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def auth_dep(authorization: Optional[str] = None):
    token = None
    if authorization and authorization.lower().startswith('bearer '):
        token = authorization.split(' ', 1)[1]
    return verify_bearer(token)

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


async def predict_with_modal(image_b64: str, image_width: int = 640, image_height: int = 480) -> List[PredictionResult]:
    """Single unified function to get predictions from Modal API or return mock predictions"""
    
    if USE_MOCK_MODAL:
        # Use mock predictions for development/testing
        logger.info("Using mock Modal predictions")
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Generate random predictions with randomized positions and sizes
        mock_predictions = [
            PredictionResult(
                class_id=0, 
                confidence=round(random.uniform(0.85, 0.98), 2), 
                label="person",
                bbox=BoundingBox(
                    x=random.randint(50, max(51, image_width // 2)), 
                    y=random.randint(30, max(31, image_height // 3)), 
                    width=random.randint(image_width // 4, image_width // 2), 
                    height=random.randint(image_height // 3, image_height // 2)
                )
            ),
            PredictionResult(
                class_id=1, 
                confidence=round(random.uniform(0.75, 0.95), 2), 
                label="car",
                bbox=BoundingBox(
                    x=random.randint(image_width // 2, max(image_width // 2 + 1, image_width - 300)), 
                    y=random.randint(image_height // 3, max(image_height // 3 + 1, image_height // 2)), 
                    width=random.randint(image_width // 3, image_width // 2), 
                    height=random.randint(image_height // 5, image_height // 3)
                )
            ),
            PredictionResult(
                class_id=2, 
                confidence=round(random.uniform(0.65, 0.88), 2), 
                label="bicycle",
                bbox=BoundingBox(
                    x=random.randint(image_width // 4, max(image_width // 4 + 1, image_width - 200)), 
                    y=random.randint(image_height // 2, max(image_height // 2 + 1, image_height - 180)), 
                    width=random.randint(image_width // 5, image_width // 3), 
                    height=random.randint(image_height // 4, image_height // 3)
                )
            )
        ]
        logger.info(f"Generated {len(mock_predictions)} mock predictions")
        return mock_predictions
    
    else:
        # Use real Modal API for production
        logger.info("Calling real Modal API")
        if not MODAL_ENDPOINT_URL:
            logger.error("Modal endpoint URL not configured for production")
            raise ValueError("Modal API not configured")
        
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{MODAL_ENDPOINT_URL}/predict",
                    json={"image": image_b64, "width": image_width, "height": image_height}
                )
                response.raise_for_status()
                
                # Parse Modal API response to PredictionResult objects
                modal_data = response.json()
                predictions = []
                
                for item in modal_data.get("predictions", []):
                    bbox = None
                    if item.get("bbox"):
                        bbox = BoundingBox(
                            x=item["bbox"]["x"],
                            y=item["bbox"]["y"], 
                            width=item["bbox"]["width"],
                            height=item["bbox"]["height"]
                        )
                    
                    predictions.append(PredictionResult(
                        class_id=item["class_id"],
                        confidence=item["confidence"],
                        label=item.get("label"),
                        bbox=bbox
                    ))
                
                logger.info(f"Received {len(predictions)} predictions from Modal API")
                return predictions
                
        except httpx.TimeoutException:
            logger.error("Modal API request timed out")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Modal API returned error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Modal API request failed: {str(e)}")
            raise


async def process_video_with_modal(video_b64: str, frame_skip: int = 10) -> List[VideoFrame]:
    """Process entire video using Modal API and return structured VideoFrame data"""
    
    if USE_MOCK_MODAL:
        # Use mock video processing for development/testing
        logger.info("Using mock Modal video processing")
        await asyncio.sleep(1.0)  # Simulate longer processing time for video
        
        # Generate mock video frames 
        mock_frames = []
        for i in range(0, 100, frame_skip):  # Mock 100 frames total
            timestamp = i * 0.033  # ~30fps
            
            # Generate random predictions for this frame
            mock_predictions = [
                PredictionResult(
                    class_id=0,
                    confidence=round(random.uniform(0.85, 0.98), 2),
                    label="person",
                    bbox=BoundingBox(
                        x=random.randint(50, 320),
                        y=random.randint(30, 160),
                        width=random.randint(160, 320),
                        height=random.randint(160, 240)
                    )
                ),
                PredictionResult(
                    class_id=1,
                    confidence=round(random.uniform(0.75, 0.95), 2),
                    label="car",
                    bbox=BoundingBox(
                        x=random.randint(320, 540),
                        y=random.randint(160, 240),
                        width=random.randint(200, 320),
                        height=random.randint(100, 160)
                    )
                )
            ]
            
            mock_frames.append(VideoFrame(
                frame_number=i,
                timestamp=timestamp,
                predictions=mock_predictions
            ))
        
        logger.info(f"Generated {len(mock_frames)} mock video frames")
        return mock_frames
    
    else:
        # Use real Modal API for production
        logger.info("Calling real Modal video processing API")
        if not MODAL_ENDPOINT_URL:
            logger.error("Modal endpoint URL not configured for production")
            raise ValueError("Modal API not configured")
        
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT * 3) as client:  # Longer timeout for video
                response = await client.post(
                    f"{MODAL_ENDPOINT_URL}/process_video",
                    json={"video": video_b64, "frame_skip": frame_skip}
                )
                response.raise_for_status()
                
                # Parse Modal API response to VideoFrame objects
                modal_data = response.json()
                video_frames = []
                
                for frame_data in modal_data.get("frames", []):
                    # Convert predictions to PredictionResult objects
                    predictions = []
                    for pred in frame_data.get("predictions", []):
                        bbox = None
                        if pred.get("bbox"):
                            bbox = BoundingBox(
                                x=pred["bbox"]["x"],
                                y=pred["bbox"]["y"],
                                width=pred["bbox"]["width"],
                                height=pred["bbox"]["height"]
                            )
                        
                        predictions.append(PredictionResult(
                            class_id=pred["class_id"],
                            confidence=pred["confidence"],
                            label=pred.get("label"),
                            bbox=bbox
                        ))
                    
                    video_frames.append(VideoFrame(
                        frame_number=frame_data["frame_number"],
                        timestamp=frame_data["timestamp"],
                        predictions=predictions
                    ))
                
                logger.info(f"Received {len(video_frames)} processed frames from Modal API")
                return video_frames
                
        except httpx.TimeoutException:
            logger.error("Modal video processing API request timed out")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Modal video processing API returned error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Modal video processing API request failed: {str(e)}")
            raise


def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            log_video_processing("Temporary file cleaned up", path=file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")


async def save_temp_video(video_data: bytes) -> str:
    """Save video data to temporary file and return path"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"video_{uuid.uuid4().hex}.mp4")
    
    log_video_processing("Saving video to temporary file", 
                        path=temp_path, 
                        size_mb=f"{len(video_data) / (1024*1024):.2f}")
    
    async with aiofiles.open(temp_path, 'wb') as f:
        await f.write(video_data)
    
    log_video_processing("Video saved to temporary file", path=temp_path)
    return temp_path


@app.post("/predict_video", response_model=VideoProcessingResponse)
async def predict_video(request: VideoProcessingRequest):
    """Process video with ML model using efficient Modal batch processing"""
    try:
        logger.info(f"Starting video processing for file: {request.filename}")
        logger.info(f"Video data length: {len(request.video_data)} characters")
        logger.info(f"Requested codec: {request.video_codec}, format: {request.output_format}")
        
        # Set default codec if not provided
        video_codec = request.video_codec or "h264"
        
        # Determine output format from request
        filename_parts = request.filename.split('.')
        input_extension = filename_parts[-1] if len(filename_parts) > 1 and filename_parts[-1] else "mp4"
        
        if request.output_format and request.output_format in ["mp4", "webm"]:
            output_format = request.output_format
        else:
            # Try to preserve input format, with fallbacks
            if input_extension.lower() in ["mp4", "m4v", "mov"]:
                output_format = "mp4"
            elif input_extension.lower() in ["webm", "mkv"]:
                output_format = "webm"
            else:
                output_format = "mp4" if video_codec in ["h264", "mp4v"] else "webm"
        
        logger.info(f"Input format: {input_extension}, Output format: {output_format}")
        
        # Use new Modal video processing approach - send entire video to Modal
        frame_skip = 10  # Process every 10th frame
        logger.info(f"Processing video with Modal (frame_skip={frame_skip})")
        
        # Process entire video using Modal API
        processed_frames = await process_video_with_modal(request.video_data, frame_skip)
        logger.info(f"Received {len(processed_frames)} processed frames from Modal")
        
        # Return response with frame predictions (no video encoding needed for client-side overlay)
        response_data = {
            "success": True,
            "output_video": "",  # Empty since we use client-side overlays
            "output_format": output_format,
            "frames": [frame.model_dump() for frame in processed_frames],
            "total_frames": len(processed_frames),
            "processed_frames": len(processed_frames)
        }
        
        # Debug: log what we're sending to frontend
        logger.info(f"Sending response with {len(processed_frames)} frames")
        if processed_frames:
            sample_frame = processed_frames[0]
            logger.info(f"Sample frame: number={sample_frame.frame_number}, timestamp={sample_frame.timestamp}, predictions={len(sample_frame.predictions)}")
            if sample_frame.predictions:
                sample_pred = sample_frame.predictions[0]
                logger.info(f"Sample prediction: label={sample_pred.label}, bbox={sample_pred.bbox}")
        
        return VideoProcessingResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {e}", exc_info=True)
        
        # Return error response instead of raising HTTPException
        return VideoProcessingResponse(
            success=False,
            error=str(e)
        )


@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing"""
    await websocket.accept()
    temp_files = []
    
    try:
        await websocket.send_json({"type": "welcome", "message": "Video processing WebSocket connected"})
        
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break
                
            data = await websocket.receive_json()
            
            if data.get("type") == "video_frame":
                try:
                    # Process single frame
                    frame_b64 = data.get("frame")
                    if not frame_b64:
                        await websocket.send_json({"type": "error", "message": "No frame data provided"})
                        continue
                    
                    # Decode frame
                    frame_data = base64.b64decode(frame_b64)
                    
                    # Get predictions
                    predictions = await predict_with_modal(frame_b64)
                    
                    # For client-side overlay, only send predictions (no frame processing)
                    processed_frame = None
                    
                    # Send result
                    await websocket.send_json({
                        "type": "frame_result",
                        "frame_number": data.get("frame_number", 0),
                        "predictions": [pred.model_dump() for pred in predictions],
                        "processed_frame": processed_frame,
                        "processing_time": time.time()
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error", 
                        "message": f"Frame processing error: {str(e)}"
                    })
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            await asyncio.sleep(0)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Clean up any temporary files
        for temp_file in temp_files:
            cleanup_temp_file(temp_file)
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), _user=Depends(auth_dep)):
    logger.info(f"Image prediction request received: {file.filename}, size: {file.size} bytes")
    
    client_key = "default"
    if not limiter.allow(client_key):
        logger.warning(f"Rate limit exceeded for client: {client_key}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    contents = await file.read()
    ok, msg = validate_image_file(file.filename, file.content_type or '', MAX_FILE_SIZE, len(contents))
    if not ok:
        logger.error(f"Image validation failed: {msg}")
        raise HTTPException(status_code=400, detail=msg)

    t0 = time.perf_counter()
    try:
        log_processing_step("Starting image processing pipeline", {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(contents)
        })
        
        # Get original image dimensions BEFORE resizing
        nparr = np.frombuffer(contents, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_img is None:
            raise ValueError("Failed to decode image for dimension detection")
        original_height, original_width = original_img.shape[:2]
        
        logger.info(f"Original image dimensions: {original_width}x{original_height}")
        
        resized = resize_image_if_needed(contents)
        image_b64 = convert_to_base64(resized)
        
        log_processing_step("Sending image for prediction")
        
        # Use original image dimensions for coordinate generation, not resized dimensions
        preds = await predict_with_modal(image_b64, image_width=original_width, image_height=original_height)
        
        elapsed = time.perf_counter() - t0
        resp = PredictionResponse(success=True, predictions=preds, processing_time=elapsed)
        
        log_request("/predict", elapsed, True, {
            "predictions": len(preds),
            "filename": file.filename,
        })
        return resp
        
    except httpx.TimeoutException:
        elapsed = time.perf_counter() - t0
        log_request("/predict", elapsed, False, {"error": "upstream_timeout"})
        return PredictionResponse(success=False, error="Upstream timeout", processing_time=elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.exception(f"Image prediction failed: {str(e)}")
        log_request("/predict", elapsed, False, {"error": str(e)})
        return PredictionResponse(success=False, error=str(e), processing_time=elapsed)


@app.get("/stream")
async def get_stream_interface():
    """Serve the streaming interface - redirects to Modal WebRTC service"""
    if USE_MOCK_MODAL:
        # Return mock streaming interface for development
        return {
            "message": "Mock WebRTC streaming server ready", 
            "websocket_url": "/stream/ws/{client_id}",
            "mode": "mock"
        }
    
    if not MODAL_ENDPOINT_URL:
        raise HTTPException(status_code=503, detail="Modal service not configured")
    
    return {
        "message": "Modal WebRTC streaming server ready", 
        "websocket_url": f"{MODAL_ENDPOINT_URL}/ws/{{client_id}}",
        "mode": "production"
    }


@app.websocket("/stream/ws/{client_id}")
async def websocket_stream_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for WebRTC signaling - proxies to Modal or provides mock"""
    await websocket.accept()
    
    if USE_MOCK_MODAL:
        # Mock WebRTC signaling for development
        try:
            await websocket.send_text(json.dumps({
                "type": "identified",
                "peer_id": f"mock-peer-{client_id[:8]}"
            }))
            
            # Mock message handling loop
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    msg_type = message.get("type")
                    
                    if msg_type == "offer":
                        # Mock WebRTC answer
                        await websocket.send_text(json.dumps({
                            "type": "answer",
                            "sdp": "mock-sdp-answer",
                            "peer_id": f"mock-peer-{client_id[:8]}"
                        }))
                    elif msg_type == "identify":
                        await websocket.send_text(json.dumps({
                            "type": "identified", 
                            "peer_id": f"mock-peer-{client_id[:8]}"
                        }))
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
                    
        except Exception as e:
            logger.error(f"Mock WebSocket error: {e}")
        finally:
            try:
                await websocket.close()
            except:
                pass
    else:
        # Proxy to Modal WebRTC service in production
        if not MODAL_ENDPOINT_URL:
            await websocket.close(code=1011, reason="Modal service not configured")
            return
            
        try:
            # Connect to Modal WebRTC service
            import websockets
            
            modal_ws_url = f"{MODAL_ENDPOINT_URL}/ws/{client_id}".replace("http://", "ws://").replace("https://", "wss://")
            
            async with websockets.connect(modal_ws_url) as modal_ws:
                # Proxy messages between client and Modal WebRTC service
                async def client_to_modal():
                    try:
                        while True:
                            data = await websocket.receive_text()
                            await modal_ws.send(data)
                    except WebSocketDisconnect:
                        pass
                
                async def modal_to_client():
                    try:
                        async for message in modal_ws:
                            await websocket.send_text(message)
                    except:
                        pass
                
                # Run both proxy directions concurrently
                await asyncio.gather(
                    client_to_modal(),
                    modal_to_client(),
                    return_exceptions=True
                )
                
        except Exception as e:
            logger.error(f"Modal WebRTC proxy error: {e}")
            await websocket.close(code=1011, reason="Failed to connect to Modal WebRTC service")


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
