# Modal GPU service for computer vision inference
import base64
import io
import os
import tempfile
import time
from typing import Dict, List, Optional

import modal

# Modal app configuration
app = modal.App("mimetic-demo")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi[all]",
    "pillow",
    "opencv-python-headless",
    "numpy",
    "torch",
    "torchvision", 
    "ultralytics",  # For YOLO models
    "pydantic"
])

@app.function(
    image=image,
    gpu="any",
    scaledown_window=300,
    timeout=3600
)
def process_image(image_b64: str, width: int = 640, height: int = 480) -> List[Dict]:
    
    '''
    Run inference on a single image
    Replace this with your actual model inference
    '''

    try:
        # Import inside function to avoid Modal deployment issues
        from PIL import Image

        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Mock prediction - replace with your actual model
        # Example with YOLO or other object detection model:
        # model = YOLO('yolov8n.pt')  # or load your custom model
        # results = model(image)
        # predictions = process_results(results, width, height)
        
        # Mock predictions for now
        predictions = [
            {
                "class_id": 0,
                "confidence": 0.95,
                "label": "detected_object",
                "bbox": {
                    "x": width * 0.2,
                    "y": height * 0.2,
                    "width": width * 0.4,
                    "height": height * 0.3
                }
            }
        ]
        
        return predictions
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

@app.function(
    image=image,
    gpu="any",
    scaledown_window=300,
    timeout=3600
)
def process_video(video_b64: str, frame_skip: int = 10) -> List[Dict]:
    '''
    Process entire video and return predictions for all frames
    This is more efficient than processing frames individually
    '''
    
    try:
        # Import inside function to avoid Modal deployment issues
        import cv2
        import numpy as np
        
        print(f"Starting video processing with frame_skip={frame_skip}")
        
        # Decode base64 video
        video_bytes = base64.b64decode(video_b64)
        
        # Save to temporary file for OpenCV processing
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name
        
        # Initialize OpenCV video capture
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            print(f"Could not open video file")
            return []
        
        # Get video properties
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video properties: {video_width}x{video_height}, fps={fps}")
        
        # Initialize model once (this is the key efficiency gain)
        # model = YOLO('yolov8n.pt')  # Load your model once here
        
        processed_frames = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame based on frame_skip
                if frame_count % frame_skip == 0:
                    timestamp = frame_count / fps if fps > 0 else frame_count * 0.033  # fallback to ~30fps
                    
                    # Run inference on this frame
                    # For now, mock predictions. Replace with:
                    # results = model(frame)
                    # frame_predictions = process_model_results(results, video_width, video_height)
                    
                    # Mock predictions with some variation
                    import random
                    frame_predictions = [
                        {
                            "class_id": 0,
                            "confidence": round(random.uniform(0.85, 0.98), 2),
                            "label": "person",
                            "bbox": {
                                "x": random.randint(50, max(51, video_width // 2)),
                                "y": random.randint(30, max(31, video_height // 3)),
                                "width": random.randint(video_width // 4, video_width // 2),
                                "height": random.randint(video_height // 3, video_height // 2)
                            }
                        },
                        {
                            "class_id": 1,
                            "confidence": round(random.uniform(0.75, 0.95), 2),
                            "label": "car",
                            "bbox": {
                                "x": random.randint(video_width // 2, max(video_width // 2 + 1, video_width - 300)),
                                "y": random.randint(video_height // 3, max(video_height // 3 + 1, video_height // 2)),
                                "width": random.randint(video_width // 3, video_width // 2),
                                "height": random.randint(video_height // 5, video_height // 3)
                            }
                        }
                    ]
                    
                    # Create frame data structure
                    frame_data = {
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "predictions": frame_predictions
                    }
                    
                    processed_frames.append(frame_data)
                
                frame_count += 1
            
        finally:
            cap.release()
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass
        
        print(f"Processed {len(processed_frames)} frames from {frame_count} total frames")
        return processed_frames
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return []

# Create FastAPI app inside function to avoid import issues
def create_web_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    # Define Pydantic models inside function
    class PredictionResult(BaseModel):
        class_id: int
        confidence: float
        label: str | None = None
        bbox: Optional[Dict[str, float]] = None

    class PredictRequest(BaseModel):
        image: str  # base64 encoded image
        width: Optional[int] = 640
        height: Optional[int] = 480
    
    class VideoProcessRequest(BaseModel):
        video: str  # base64 encoded video
        frame_skip: Optional[int] = 10
    
    web_app = FastAPI(title="Modal CV Service")
    
    # Add CORS middleware to handle OPTIONS requests
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://mimetic-demo*",
            "http://localhost:3000",  # for development
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/health")
    async def health_http():
        """HTTP health check"""
        return {"status": "ok", "model": "yolo", "gpu": True}

    @web_app.post("/predict")
    async def predict_http(req: PredictRequest):
        """HTTP prediction endpoint for single images"""
        try:
            predictions = process_image.remote(req.image, req.width or 640, req.height or 480)
            return {"success": True, "predictions": predictions}
        except Exception as e:
            return {"success": False, "error": str(e), "predictions": []}
    
    @web_app.post("/process_video")
    async def process_video_http(req: VideoProcessRequest):
        """HTTP endpoint for processing entire videos"""
        try:
            processed_frames = process_video.remote(req.video, req.frame_skip or 10)
            return {"success": True, "frames": processed_frames}
        except Exception as e:
            return {"success": False, "error": str(e), "frames": []}
    
    return web_app

@app.function(image=image)
@modal.asgi_app()
def web():
    """Expose the FastAPI app as a Modal ASGI app"""
    return create_web_app()

if __name__ == "__main__":
    # For local development
    print("Modal CV Inference Service")
    print("Deploy with: modal deploy modal_service.py")
