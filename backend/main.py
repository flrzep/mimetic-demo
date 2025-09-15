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
from models import (BoundingBox, HealthResponse, PredictionRequest,
                    PredictionResponse, PredictionResult, VideoFrame,
                    VideoProcessingRequest, VideoProcessingResponse)
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
MODAL_API_KEY = os.getenv("MODAL_API_KEY", "")

# Dynamic CORS based on environment
DEFAULT_CORS = "https://memetic-demo-*.vercel.app" if IS_PRODUCTION else "http://localhost:3000"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", DEFAULT_CORS).split(",")

def is_cors_allowed(origin: str, allowed_patterns: List[str]) -> bool:
    """Check if an origin matches any of the allowed CORS patterns (supports wildcards)"""
    if not origin:
        return False
    
    for pattern in allowed_patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
            
        # Exact match
        if origin == pattern:
            return True
            
        # Wildcard pattern matching for Vercel URLs
        if '*' in pattern:
            # Convert wildcard pattern to regex
            regex_pattern = re.escape(pattern).replace(r'\*', '.*')
            if re.fullmatch(regex_pattern, origin):
                return True
    
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
_jwks_cache = TTLCache(maxsize=1, ttl=3600)

# Log environment info
logger.info(f"Environment: {'Production' if IS_PRODUCTION else 'Development'}")
logger.info(f"Debug mode: {DEBUG_MODE}")
logger.info(f"Use mock Modal: {USE_MOCK_MODAL}")
logger.info(f"CORS patterns: {CORS_PATTERNS}")

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


async def forward_to_modal(image_b64: str, video_width: int = 640, video_height: int = 480) -> list[PredictionResult]:
    """Forward image to Modal API or return mock predictions based on environment"""
    
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
                    x=random.randint(50, max(51, video_width // 2)), 
                    y=random.randint(30, max(31, video_height // 3)), 
                    width=random.randint(80, min(150, video_width // 4)), 
                    height=random.randint(150, min(250, video_height // 2))
                )
            ),
            PredictionResult(
                class_id=1, 
                confidence=round(random.uniform(0.75, 0.95), 2), 
                label="car",
                bbox=BoundingBox(
                    x=random.randint(video_width // 2, max(video_width // 2 + 1, video_width - 200)), 
                    y=random.randint(video_height // 3, max(video_height // 3 + 1, video_height // 2)), 
                    width=random.randint(120, min(200, video_width // 3)), 
                    height=random.randint(80, min(140, video_height // 3))
                )
            ),
            PredictionResult(
                class_id=2, 
                confidence=round(random.uniform(0.65, 0.88), 2), 
                label="bicycle",
                bbox=BoundingBox(
                    x=random.randint(min(video_width // 4, 21), max(21, video_width // 3)), 
                    y=random.randint(video_height // 2, max(video_height // 2 + 1, video_height - 180)), 
                    width=random.randint(60, min(120, video_width // 5)), 
                    height=random.randint(80, min(160, video_height // 3))
                )
            )
        ]
        logger.info(f"Generated {len(mock_predictions)} mock predictions")
        return mock_predictions
    
    else:
        # Use real Modal API for production
        logger.info("Calling real Modal API")
        if not MODAL_ENDPOINT_URL or not MODAL_API_KEY:
            logger.error("Modal API credentials not configured for production")
            raise ValueError("Modal API not configured")
        
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{MODAL_ENDPOINT_URL}/predict",
                    json={"image": image_b64},
                    headers={"Authorization": f"Bearer {MODAL_API_KEY}"}
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


def add_simple_image_annotations(image_data: bytes, predictions: list[PredictionResult]) -> str:
    """Simple image annotation for image endpoint only (not used for video)"""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    height, width = img.shape[:2]

    for pred in predictions:
        label = f"{pred.label or f'Class {pred.class_id}'}: {pred.confidence:.2f}"
        
        if pred.bbox:
            # Draw bounding box
            x = max(0, min(pred.bbox.x, width - 1))
            y = max(0, min(pred.bbox.y, height - 1))
            w = max(1, min(pred.bbox.width, width - x))
            h = max(1, min(pred.bbox.height, height - y))
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
            label_y = max(y - 10, label_size[1] + 5)
            cv2.rectangle(img, (x, label_y - label_size[1] - 5), (x + label_size[0], label_y + 5), (0, 255, 0), -1)
            cv2.putText(img, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')


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


async def collect_processed_frames(video_path: str) -> List[VideoFrame]:
    """Collect all processed frames from video for batch processing"""
    processed_frames = []
    
    logger.info(f"Starting frame collection from video: {video_path}")
    
    # Get actual video dimensions from the file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        raise ValueError("Could not open video file")
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    logger.info(f"Video dimensions: {video_width}x{video_height}")
    
    # Create consistent but slightly moving predictions per video
    # Use video path as seed for reproducible randomness
    import hashlib
    video_hash = hashlib.md5(video_path.encode()).hexdigest()
    base_seed = int(video_hash[:8], 16)
    
    # Generate initial positions
    random.seed(base_seed)
    
    # Base positions for each object (consistent per video)
    base_positions = {
        'person': {
            'x': random.randint(50, video_width // 2),
            'y': random.randint(30, video_height // 3),
            'width': random.randint(80, 150) * video_width // 640,
            'height': random.randint(150, 250) * video_height // 480
        },
        'car': {
            'x': random.randint(video_width // 2, video_width - 200),
            'y': random.randint(video_height // 3, video_height // 2),
            'width': random.randint(120, 200) * video_width // 640,
            'height': random.randint(80, 140) * video_height // 480
        },
        'bicycle': {
            'x': random.randint(20, video_width // 3),
            'y': random.randint(video_height // 2, video_height - 180),
            'width': random.randint(60, 120) * video_width // 640,
            'height': random.randint(80, 160) * video_height // 480
        }
    }
    
    async for frame_num, frame, timestamp in process_video_frames(video_path):
        # Convert frame to bytes for prediction
        _, frame_buffer = cv2.imencode('.jpg', frame)
        frame_bytes = frame_buffer.tobytes()
        frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
        
        # Create slightly moving predictions based on frame number
        # Small random movement (±5 pixels) from base positions
        random.seed(base_seed + frame_num)  # Deterministic but different per frame

        range_x = video_width // 100    # Move left/right over time
        range_y = video_height // 100    # Move up/down over time

        predictions = [
            PredictionResult(
                class_id=0,
                confidence=round(0.90 + random.uniform(-0.05, 0.05), 2),
                label="person",
                bbox=BoundingBox(
                    x=max(0, base_positions['person']['x'] + random.randint(-range_x, range_x)),
                    y=max(0, base_positions['person']['y'] + random.randint(-range_y, range_y)),
                    width=base_positions['person']['width'] + random.randint(-3, 3),
                    height=base_positions['person']['height'] + random.randint(-3, 3)
                )
            ),
            PredictionResult(
                class_id=1,
                confidence=round(0.85 + random.uniform(-0.05, 0.05), 2),
                label="car",
                bbox=BoundingBox(
                    x=max(0, base_positions['car']['x'] + random.randint(-range_x, range_x)),
                    y=max(0, base_positions['car']['y'] + random.randint(-range_y, range_y)),
                    width=base_positions['car']['width'] + random.randint(-3, 3),
                    height=base_positions['car']['height'] + random.randint(-3, 3)
                )
            ),
            PredictionResult(
                class_id=2,
                confidence=round(0.78 + random.uniform(-0.05, 0.05), 2),
                label="bicycle",
                bbox=BoundingBox(
                    x=max(0, base_positions['bicycle']['x'] + random.randint(-range_x, range_x)),
                    y=max(0, base_positions['bicycle']['y'] + random.randint(-range_y, range_y)),
                    width=base_positions['bicycle']['width'] + random.randint(-3, 3),
                    height=base_positions['bicycle']['height'] + random.randint(-3, 3)
                )
            )
        ]
        
        # Create VideoFrame object
        video_frame = VideoFrame(
            frame_number=frame_num,
            timestamp=timestamp,
            predictions=predictions
        )
        processed_frames.append(video_frame)
    
    total_predictions = sum(len(frame.predictions) for frame in processed_frames)
    total_bboxes = sum(sum(1 for p in frame.predictions if p.bbox) for frame in processed_frames)
    logger.info(f"Collected {len(processed_frames)} frames with {total_predictions} total predictions, {total_bboxes} with bboxes")
    
    return processed_frames

def test_codec_availability(fourcc, output_path, fps, frame_size):
    """Test if a codec is available for video writing"""
    try:
        test_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        is_opened = test_writer.isOpened()
        test_writer.release()
        return is_opened
    except Exception:
        return False

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            log_video_processing("Temporary file cleaned up", path=file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")


async def process_video_frames(video_path: str, frame_interval: int = 1, max_frames: Optional[int] = None) -> AsyncGenerator[Tuple[int, np.ndarray, float], None]:
    """Process video frames and yield frame data"""
    log_video_processing("Starting video frame processing", 
                        path=video_path, 
                        frame_interval=frame_interval, 
                        max_frames=max_frames)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    log_video_processing("Video properties detected", 
                        fps=f"{fps:.2f}", 
                        total_frames=total_frames, 
                        duration_sec=f"{duration:.2f}")
    
    frame_count = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                yield frame_count, frame, timestamp
                processed_count += 1
                
                if max_frames and processed_count >= max_frames:
                    log_video_processing("Reached max frames limit", 
                                       max_frames=max_frames, 
                                       processed=processed_count)
                    break
            
            frame_count += 1
    finally:
        cap.release()
        log_video_processing("Frame processing completed", 
                           total_frames=frame_count, 
                           processed_frames=processed_count)


async def create_output_video(original_path: str, processed_frames: List[VideoFrame], 
                             video_codec: str = "h264", output_format: str = "mp4") -> str:
    """Return None - client-side overlay only needs predictions, not video data"""
    
    logger.info("Using client-side overlay approach - no video processing needed")
    logger.info("Frontend will use original video file with Canvas overlay for predictions")
    
    # Return empty string since we don't need to send video data back
    # The frontend already has the original video file
    return ""


@app.post("/predict_video", response_model=VideoProcessingResponse)
async def predict_video(request: VideoProcessingRequest):
    """Process video with ML model and return annotated video"""
    try:
        logger.info(f"Starting video processing for file: {request.filename}")
        logger.info(f"Video data length: {len(request.video_data)} characters")
        logger.info(f"Requested codec: {request.video_codec}, format: {request.output_format}")
        
        # Set default codec if not provided
        video_codec = request.video_codec or "h264"
        
        # Decode video data
        video_data = base64.b64decode(request.video_data)
        
        # Create temporary input file with appropriate extension
        filename_parts = request.filename.split('.')
        input_extension = filename_parts[-1] if len(filename_parts) > 1 and filename_parts[-1] else "mp4"
        temp_input = os.path.join(tempfile.gettempdir(), f"input_{uuid.uuid4().hex}.{input_extension}")
        
        # Determine output format - preserve input format when possible
        if request.output_format and request.output_format in ["mp4", "webm"]:
            # Use explicitly requested format
            output_format = request.output_format
        else:
            # Try to preserve input format, with fallbacks
            if input_extension.lower() in ["mp4", "m4v", "mov"]:
                output_format = "mp4"
            elif input_extension.lower() in ["webm", "mkv"]:
                output_format = "webm"
            else:
                # Default based on codec for unknown extensions
                output_format = "mp4" if video_codec in ["h264", "mp4v"] else "webm"
        
        logger.info(f"Input format: {input_extension}, Output format: {output_format}")
        
        # Write video data to temporary file
        async with aiofiles.open(temp_input, 'wb') as f:
            await f.write(video_data)
        
        try:
            logger.info(f"Starting frame processing for video: {temp_input}")
            # Process video frames
            processed_frames = await collect_processed_frames(temp_input)
            logger.info(f"Processed {len(processed_frames)} frames")
            
            # Create output video with annotations
            logger.info(f"Creating output video with codec: {video_codec}, format: {output_format}")
            output_video_b64 = await create_output_video(temp_input, processed_frames, video_codec, output_format)
            logger.info(f"Output video created, base64 length: {len(output_video_b64)}")
            
            # Return enhanced response with frame predictions
            response_data = {
                "success": True,
                "output_video": output_video_b64,
                "output_format": output_format,
                "frames": [frame.model_dump() for frame in processed_frames],  # Include frame predictions!
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
        
        finally:
            cleanup_temp_file(temp_input)
    
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
                    predictions = await forward_to_modal(frame_b64)
                    
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
        
        resized = resize_image_if_needed(contents)
        image_b64 = convert_to_base64(resized)
        
        log_processing_step("Sending image for prediction")
        preds = await forward_to_modal(image_b64)
        
        log_processing_step("Adding predictions to image", {"prediction_count": len(preds)})
        image_response = add_simple_image_annotations(resized, preds)

        elapsed = time.perf_counter() - t0
        resp = PredictionResponse(success=True, predictions=preds, processing_time=elapsed, image=image_response)
        
        log_request("/predict", elapsed, True, {
            "predictions": len(preds),
            "filename": file.filename,
            "output_size_kb": f"{len(image_response)/1024:.1f}"
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


@app.post("/predict-base64", response_model=PredictionResponse)
async def predict_base64(req: PredictionRequest, _user=Depends(auth_dep)):
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
