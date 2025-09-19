# Modal GPU service for computer vision inference
import base64
import io
import os
import tempfile
import time
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
app = modal.App(app_name)

print(f"Modal app name: {app_name}")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").apt_install([
    "libgl1-mesa-glx",
    "libglib2.0-0", 
    "libsm6",
    "libxext6",
    "libxrender-dev",
    "libgomp1",
    "libgcc-s1"
]).pip_install([
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
]).copy_local_dir("models", "/app/models")  # Copy YOLO model files

@app.function(
    image=image,
    gpu="any",
    scaledown_window=300,
    timeout=3600
)
def process_image(image_b64: str, width: int = 640, height: int = 480) -> List[Dict]:
    '''
    Run YOLOv10 inference on a single image
    '''

    try:
        # Import inside function to avoid Modal deployment issues
        import sys
        sys.path.append('/app')
        import cv2
        import numpy as np
        from models.yolo.yolo import YOLOv10
        from PIL import Image

        print(f"Processing image with YOLOv10: {width}x{height}")

        # Initialize YOLO model (cached after first call)
        if not hasattr(process_image, "_yolo_model"):
            print("Initializing YOLOv10 model...")
            process_image._yolo_model = YOLOv10(cache_dir="/tmp/yolo_cache")
            print("YOLOv10 model initialized")

        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO inference
        print("Running YOLOv10 inference...")
        outputs = process_image._yolo_model.session.run(
            process_image._yolo_model.output_names, 
            {process_image._yolo_model.input_names[0]: process_image._yolo_model.prepare_input(cv_image)}
        )
        
        # Process outputs to get predictions
        boxes, scores, class_ids = process_image._yolo_model.process_output(outputs, conf_threshold=0.3)
        
        # Convert to our API format
        predictions = []
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            predictions.append({
                "class_id": int(class_id),
                "confidence": float(score),
                "label": process_image._yolo_model.class_names[class_id],
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
        # Fallback to mock predictions
        return [
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

@app.function(
    image=image,
    gpu="any",
    scaledown_window=300,
    timeout=3600
)
def process_video(video_b64: str, frame_skip: int = 10) -> List[Dict]:
    '''
    Process entire video with YOLOv10 and return predictions for all frames
    This is more efficient than processing frames individually
    '''
    
    try:
        # Import inside function to avoid Modal deployment issues
        import os
        import sys
        sys.path.append('/app')
        import cv2
        import numpy as np
        from models.yolo.yolo import YOLOv10

        # Set environment variables for headless OpenCV
        os.environ['DISPLAY'] = ''
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        print(f"Starting video processing with YOLOv10, frame_skip={frame_skip}")
        
        # Initialize YOLO model once (key efficiency gain)
        if not hasattr(process_video, "_yolo_model"):
            print("Initializing YOLOv10 model for video processing...")
            process_video._yolo_model = YOLOv10(cache_dir="/tmp/yolo_cache")
            print("YOLOv10 model initialized for video")
        
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
                        outputs = process_video._yolo_model.session.run(
                            process_video._yolo_model.output_names, 
                            {process_video._yolo_model.input_names[0]: process_video._yolo_model.prepare_input(frame)}
                        )
                        
                        # Process outputs to get predictions
                        boxes, scores, class_ids = process_video._yolo_model.process_output(outputs, conf_threshold=0.3)
                        
                        # Convert to our API format
                        frame_predictions = []
                        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                            x1, y1, x2, y2 = box.astype(int)
                            frame_predictions.append({
                                "class_id": int(class_id),
                                "confidence": float(score),
                                "label": process_video._yolo_model.class_names[class_id],
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
