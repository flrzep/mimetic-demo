# Modal GPU service for computer vision inference
import base64
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import modal


# Determine app name based on environment/branch
def get_app_name():
    # Check for GitHub Actions environment variable
    github_ref = os.getenv("GITHUB_REF", "")
    if github_ref:
        # Extract branch name from refs/heads/branch-name
        if github_ref.startswith("refs/heads/"):
            branch = github_ref[11:]  # Remove "refs/heads/"
            if branch == "main":
                return "mimetic-demo"
            else:
                # Sanitize branch name for Modal (replace special chars with hyphens)
                safe_branch = branch.replace("/", "-").replace("_", "-").replace(".", "-")
                return f"mimetic-demo-{safe_branch}"
    
    # Local development or fallback
    return os.getenv("MODAL_APP_NAME", "mimetic-demo-dev")

# Modal app configuration with dynamic name
app_name = get_app_name()

print(f"Modal app name: {app_name}")

# Define the Modal image with required dependencies (recommended approach for Modal)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libgl1-mesa-dev",
        "libglib2.0-0", 
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libgcc-s1"
    ])
    .pip_install([
        "fastapi[all]",
        "pillow",
        "opencv-python-headless",
        "numpy",
        "torch",
        "torchvision", 
        "onnxruntime",
        "transformers",
        "datasets",
        "accelerate",
        "timm",
        "huggingface-hub",
        "pydantic"
    ])
)

# Create the Modal app with branch-based naming
app = modal.App(app_name, image=image)

# Create volumes for model caching and weights
model_cache_volume = modal.Volume.from_name("yolo-models", create_if_missing=True)
model_weights_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

MODEL_DIR = Path("/models")
CACHE_DIR = Path("/cache")

# Pre-download YOLO model weights function
@app.function(
    image=image,
    volumes={
        "/cache": model_cache_volume,
        "/models": model_weights_volume
    },
    timeout=3600
)
def download_model_weights():
    """Pre-download and cache YOLO model weights"""
    import shutil

    from huggingface_hub import hf_hub_download
    
    print("Downloading YOLO model weights...")
    
    # Download to cache first
    cache_path = CACHE_DIR / "yolo"
    cache_path.mkdir(parents=True, exist_ok=True)
    
    model_file = hf_hub_download(
        repo_id="onnx-community/yolov10n",
        filename="onnx/model.onnx",
        cache_dir=str(cache_path),
    )
    
    # Copy to persistent model weights volume
    models_yolo_dir = MODEL_DIR / "yolo"
    models_yolo_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the downloaded model to the weights volume
    destination = models_yolo_dir / "yolov10n.onnx"
    shutil.copy2(model_file, destination)
    
    # Copy the class names file from the local models directory
    local_classes_file = Path("models/yolo/yolo_classes.txt")
    if local_classes_file.exists():
        shutil.copy2(local_classes_file, models_yolo_dir / "yolo_classes.txt")
        print(f"Class names copied from {local_classes_file}")
    else:
        print("Warning: models/yolo/yolo_classes.txt not found, model will use fallback classes")
    
    print(f"Model weights saved to {destination}")
    print(f"Class names saved to {models_yolo_dir / 'yolo_classes.txt'}")
    return str(destination)

@app.function(
    image=image,
    volumes={
        "/cache": model_cache_volume,
        "/models": model_weights_volume
    }
)
def list_model_weights():
    """List all model weights currently stored in the Modal volumes"""
    import os
    from pathlib import Path
    
    print("Checking Modal volumes for model weights...")
    
    results = {
        "models_volume": {},
        "cache_volume": {}
    }
    
    # Check /models volume
    models_path = Path("/models")
    print(f"Checking /models volume at {models_path}")
    if models_path.exists():
        print(f"/models exists, listing contents...")
        for model_dir in models_path.iterdir():
            print(f"Found directory: {model_dir}")
            if model_dir.is_dir():
                files = []
                total_size = 0
                for file in model_dir.rglob("*"):
                    if file.is_file():
                        size = file.stat().st_size
                        files.append({
                            "name": file.name,
                            "path": str(file.relative_to(models_path)),
                            "size_mb": round(size / (1024 * 1024), 2)
                        })
                        total_size += size
                        print(f"  File: {file.name} ({round(size / (1024 * 1024), 2)} MB)")
                
                results["models_volume"][model_dir.name] = {
                    "files": files,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }
    else:
        print("/models does not exist")
    
    # Check /cache volume
    cache_path = Path("/cache")
    print(f"Checking /cache volume at {cache_path}")
    if cache_path.exists():
        print(f"/cache exists, listing contents...")
        for item in cache_path.iterdir():
            print(f"Found in cache: {item}")
            if item.is_dir():
                files = []
                total_size = 0
                for file in item.rglob("*"):
                    if file.is_file():
                        size = file.stat().st_size
                        files.append({
                            "name": file.name,
                            "path": str(file.relative_to(cache_path)),
                            "size_mb": round(size / (1024 * 1024), 2)
                        })
                        total_size += size
                
                results["cache_volume"][item.name] = {
                    "files": files,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }
    else:
        print("/cache does not exist")
    
    print(f"Final results: {results}")
    return results

# Global variable to store the model instance
model = None

def get_model():
    """Initialize model once and reuse across function calls"""
    global model
    if model is None:
        print("Loading model...")
        
        import onnxruntime
        onnxruntime.preload_dlls()
        
        # Import YOLOv10 from our local yolo_model.py file
        import os
        import sys

        # Import and use the local model implementation
        from import_model import YOLOv10
        
        cache_path = "/cache/yolo"
        os.makedirs(cache_path, exist_ok=True)
        
        # Initialize the model
        model = YOLOv10()
        print("Model loaded successfully")
        
        print("Model ready")
    return model

@app.function(
    image=image,
    gpu="any",
    volumes={
        "/cache": model_cache_volume,
        "/models": model_weights_volume
    },
    scaledown_window=300,
    timeout=3600
)
def process_image(image_b64: str, width: int = 640, height: int = 480) -> List[Dict]:
    '''
    Run object detection inference on a single image
    '''

    try:
        # Import inside function to avoid Modal deployment issues
        import cv2
        import numpy as np
        from PIL import Image

        print(f"Processing image with model: {width}x{height}")

        # Get the shared model instance
        model = get_model()

        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO inference
        print("Running YOLOv10 inference...")
        results = model.predict(cv_image)
        
        # Convert to our API format
        predictions = []
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            predictions.append({
                "class_id": model.class_names.index(result["class"]),
                "confidence": result["confidence"],
                "label": result["class"],
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                }
            })
        
        print(f"YOLOv10 detected {len(predictions)} objects")
        return predictions
        
    except Exception as e:
        print(f"Error processing image with YOLOv10: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception instead of returning mock data
        raise e

@app.function(
    image=image,
    gpu="any",
    volumes={
        "/cache": model_cache_volume,
        "/models": model_weights_volume
    },
    scaledown_window=300,
    timeout=3600
)
def process_video(video_b64: str, frame_skip: int = 10) -> List[Dict]:
    '''
    Process entire video with object detection and return predictions for all frames
    This is more efficient than processing frames individually
    '''
    
    try:
        # Import inside function to avoid Modal deployment issues
        import os

        import cv2
        import numpy as np

        # Set environment variables for headless OpenCV
        os.environ['DISPLAY'] = ''
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        print(f"Starting video processing with model, frame_skip={frame_skip}")
        
        # Get the shared model instance
        model = get_model()
        
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
                    
                    # Run YOLOv10 inference on this frame
                    print(f"Processing frame {frame_count} at timestamp {timestamp:.2f}s")
                    
                    try:
                        # Run YOLO inference
                        results = model.predict(frame)
                        
                        # Convert to our API format
                        frame_predictions = []
                        for result in results:
                            x1, y1, x2, y2 = result["bbox"]
                            frame_predictions.append({
                                "class_id": model.class_names.index(result["class"]),
                                "confidence": result["confidence"],
                                "label": result["class"],
                                "bbox": {
                                    "x": float(x1),
                                    "y": float(y1),
                                    "width": float(x2 - x1),
                                    "height": float(y2 - y1)
                                }
                            })
                        
                        print(f"Frame {frame_count}: detected {len(frame_predictions)} objects")
                        
                    except Exception as frame_error:
                        print(f"Error processing frame {frame_count}: {frame_error}")
                        frame_predictions = []  # Empty predictions for failed frames
                    
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
        import traceback
        traceback.print_exc()
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
    print(f"App name: {app_name}")
    print("Deploy with: modal deploy modal_service.py")
    print("For local dev with custom name: MODAL_APP_NAME=my-test-app modal deploy modal_service.py")
    print()
    print("Available commands:")
    print("- Download model weights: modal run modal_service.py::download_model_weights")
    print("- Process single image: modal run modal_service.py::process_image --image-b64 <base64_string>")
    print("- Start web server: modal serve modal_service.py")
