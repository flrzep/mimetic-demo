# Modal GPU service for computer vision inference
import base64
import io
import os
import sys
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
            if branch == "main" or branch.startswith("deploy/"):
                return "mimetic-demo"
            else:
                # Sanitize branch name for Modal (replace special chars with hyphens)
                safe_branch = branch.replace("/", "-").replace("_", "-").replace(".", "-")
                return f"mimetic-demo-{safe_branch}"
    
    # Local development or fallback
    return os.getenv("MODAL_APP_NAME", "mimetic-demo-dev")

# Modal app configuration with dynamic name
app_name = get_app_name()

# GPU configuration from environment variable
GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "any")  # Default to "any" if not set

print(f"Modal app name: {app_name}")
print(f"Modal GPU type: {GPU_TYPE}")

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
        "pyyaml",
        "transformers",
        "datasets",
        "accelerate",
        "timm",
        "huggingface-hub",
        "pydantic"
    ])
    .add_local_python_source("import_model")
    .add_local_dir("models", remote_path="/app/models")
    .add_local_file("model_config.json", remote_path="/app/model_config.json")
)

# Create the Modal app with branch-based naming
app = modal.App(app_name, image=image)

# Create volume for model storage
model_volume = modal.Volume.from_name("mimetic-models", create_if_missing=True)

MODEL_DIR = Path("/models")

@app.function(
    image=image,
    volumes={
        "/models": model_volume
    }
)
def list_model_weights():
    """List all model weights currently stored in the Modal volume"""
    from pathlib import Path
    
    print("Checking Modal volume for model weights...")
    
    results = {"models": {}}
    
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
                
                results["models"][model_dir.name] = {
                    "files": files,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }
    else:
        print("/models does not exist")
    
    print(f"Final results: {results}")
    return results

# Global variable to store model instances
models = {}

def get_model(model_name: str = "yolo"):
    """Initialize model once and reuse across function calls"""
    global models
    if model_name not in models:
        print(f"Loading model: {model_name}")
        
        import onnxruntime
        onnxruntime.preload_dlls()
        
        # Load model config to get file paths
        import json
        import yaml
        model_config = {}
        try:
            with open("/app/model_config.json", "r") as f:
                config = json.load(f)
                model_config = config.get("models", {}).get(model_name, {})
        except Exception as e:
            print(f"Warning: Could not load model config: {e}")
            
        # Get model file information from config
        files_info = model_config.get("files", {})
        model_type = model_config.get("model_type", "pytorch")
        category = model_config.get("category", "unknown")
        folder = files_info.get("folder", model_name)
        weights_file = files_info.get("weights")
        classes_file = files_info.get("classes")
        
        print(f"Model config: folder={folder}, weights={weights_file}, type={model_type}")

        # Optional: load YAML classes/keypoints metadata if provided
        class_names = None
        keypoint_names = None
        if classes_file and classes_file.endswith((".yml", ".yaml")):
            try:
                with open(os.path.join("/models", folder, classes_file), "r") as ymlf:
                    ydata = yaml.safe_load(ymlf) or {}
                    # Accept common keys
                    class_names = ydata.get("classes") or ydata.get("labels")
                    keypoint_names = ydata.get("keypoints") or ydata.get("kp")
                    print(f"Loaded YAML class metadata: classes={bool(class_names)}, keypoints={bool(keypoint_names)}")
            except Exception as yerr:
                print(f"Warning: could not parse YAML classes file: {yerr}")
        
        # Load different models based on model_name and type
        if model_name in ["yolo", "yolo_onnx"]:
            from import_model import YOLOv10
            # Pass config information to the model
            models[model_name] = YOLOv10(
                model_folder=folder,
                weights_file=weights_file,
                classes_file=classes_file,
                model_type=model_type
            )
            # Attach optional metadata if available
            if class_names is not None:
                try:
                    models[model_name].class_names = class_names
                except Exception:
                    pass
        elif model_name == "efficientnet":
            # Placeholder for EfficientNet model loading
            print(f"Model {model_name} not yet implemented, using YOLO fallback")
            from import_model import YOLOv10
            models[model_name] = YOLOv10()
        elif model_name == "keypoint_rcnn":
            # Placeholder for Keypoint R-CNN model loading
            print(f"Model {model_name} not yet implemented, using YOLO fallback")
            from import_model import YOLOv10
            models[model_name] = YOLOv10()
        elif category == "keypoint_detection" or model_name.startswith("keypoint_"):
            # Generic ONNX keypoint model loader placeholder
            # Tries to use import_model if a Keypoints class exists; otherwise raise a clear error
            try:
                from import_model import KeypointsONNX  # type: ignore
                models[model_name] = KeypointsONNX(
                    model_folder=folder,
                    weights_file=weights_file,
                    classes_file=classes_file
                )
                # Attach metadata if present
                if class_names is not None:
                    try:
                        models[model_name].class_names = class_names
                    except Exception:
                        pass
                if keypoint_names is not None:
                    try:
                        models[model_name].keypoint_names = keypoint_names
                    except Exception:
                        pass
            except Exception as e:
                print(f"Keypoint model loader not available: {e}")
                raise RuntimeError(
                    "Keypoint ONNX model support is not implemented in import_model. "
                    "Please add a KeypointsONNX class to import_model.py to enable keypoint inference."
                )
        else:
            print(f"Unknown model {model_name}, falling back to YOLO")
            from import_model import YOLOv10
            models[model_name] = YOLOv10()
            
        print(f"Model {model_name} loaded successfully")
        
    return models[model_name]

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/models": model_volume
    },
    scaledown_window=300,
    timeout=3600
)
def process_image(image_b64: str, width: int = 640, height: int = 480, model_name: str = "yolo") -> List[Dict]:
    '''
    Run object detection inference on a single image
    '''

    try:
        # Import inside function to avoid Modal deployment issues
        import cv2
        import numpy as np
        from PIL import Image

        print(f"Processing image with model '{model_name}': {width}x{height}")

        # Get the specified model instance
        model = get_model(model_name)

        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        print("Running inference...")
        results = model.predict(cv_image)
        
        # Convert to our API format
        predictions = []
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            pred = {
                "class_id": model.class_names.index(result["class"]),
                "confidence": result["confidence"],
                "label": result["class"],
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                }
            }
            # Optional keypoints passthrough if model provides them
            if "keypoints" in result and isinstance(result["keypoints"], (list, tuple)):
                # Expect a list of [x, y, score?] or dicts; normalize to dicts
                kps_out = []
                for idx, kp in enumerate(result["keypoints"]):
                    if isinstance(kp, dict):
                        kps_out.append({
                            "x": float(kp.get("x", 0.0)),
                            "y": float(kp.get("y", 0.0)),
                            **({"score": float(kp["score"]) } if "score" in kp else {})
                        })
                    elif isinstance(kp, (list, tuple)):
                        item = {"x": float(kp[0]), "y": float(kp[1])}
                        if len(kp) > 2:
                            item["score"] = float(kp[2])
                        kps_out.append(item)
                if kps_out:
                    pred["keypoints"] = kps_out
            predictions.append(pred)
        
        print(f"Detected {len(predictions)} objects")
        return predictions
        
    except Exception as e:
        print(f"Error processing image with YOLOv10: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception instead of returning mock data
        raise e

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/models": model_volume
    },
    scaledown_window=300,
    timeout=3600
)
def process_video(video_b64: str, frame_skip: int = 10, model_name: str = "yolo") -> List[Dict]:
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
        model = get_model(model_name)
        
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
                    
                    # Run inference on this frame
                    print(f"Processing frame {frame_count} at timestamp {timestamp:.2f}s")
                    
                    try:
                        # Run inference
                        results = model.predict(frame)
                        
                        # Convert to our API format
                        frame_predictions = []
                        for result in results:
                            x1, y1, x2, y2 = result["bbox"]
                            pred = {
                                "class_id": model.class_names.index(result["class"]),
                                "confidence": result["confidence"],
                                "label": result["class"],
                                "bbox": {
                                    "x": float(x1),
                                    "y": float(y1),
                                    "width": float(x2 - x1),
                                    "height": float(y2 - y1)
                                }
                            }
                            # Optional keypoints passthrough
                            if "keypoints" in result and isinstance(result["keypoints"], (list, tuple)):
                                kps_out = []
                                for idx, kp in enumerate(result["keypoints"]):
                                    if isinstance(kp, dict):
                                        kps_out.append({
                                            "x": float(kp.get("x", 0.0)),
                                            "y": float(kp.get("y", 0.0)),
                                            **({"score": float(kp["score"]) } if "score" in kp else {})
                                        })
                                    elif isinstance(kp, (list, tuple)):
                                        item = {"x": float(kp[0]), "y": float(kp[1])}
                                        if len(kp) > 2:
                                            item["score"] = float(kp[2])
                                        kps_out.append(item)
                                if kps_out:
                                    pred["keypoints"] = kps_out
                            frame_predictions.append(pred)
                        
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
        # Optional keypoints: list of {x, y, score?}
        keypoints: Optional[List[Dict[str, float]]] = None

    class PredictRequest(BaseModel):
        image: str  # base64 encoded image
        width: Optional[int] = 640
        height: Optional[int] = 480
        model: Optional[str] = "yolo"  # Default to YOLO for backward compatibility
    
    class VideoProcessRequest(BaseModel):
        video: str  # base64 encoded video
        frame_skip: Optional[int] = 10
        model: Optional[str] = "yolo"  # Default to YOLO for backward compatibility
    
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

    @web_app.get("/models")
    async def get_available_models():
        """Get list of available models with descriptions"""
        try:
            import json
            
            # Load model configuration
            with open("/app/model_config.json", "r") as f:
                config = json.load(f)
            
            model_metadata = config["models"]
            categories = config["categories"]
            
            # Get the actual models from storage
            stored_models = list_model_weights.remote()
            
            # Build response with actual available models
            available_models = []
            for model_id, model_info in stored_models.get("models", {}).items():
                metadata = model_metadata.get(model_id, {
                    "name": model_id.capitalize(),
                    "description": f"Custom {model_id} model",
                    "category": "unknown",
                    "recommended": False,
                    "input_types": ["image"],
                    "output_format": "predictions"
                })
                
                # Get category info
                category_info = categories.get(metadata["category"], {})
                
                available_models.append({
                    "id": model_id,
                    "name": metadata["name"],
                    "description": metadata["description"],
                    "category": metadata["category"],
                    "category_info": {
                        "name": category_info.get("name", metadata["category"].capitalize()),
                        "description": category_info.get("description", ""),
                        "color": category_info.get("color", "gray"),
                        "icon": category_info.get("icon", "circle")
                    },
                    "recommended": metadata.get("recommended", False),
                    "input_types": metadata.get("input_types", ["image"]),
                    "output_format": metadata.get("output_format", "predictions"),
                    "performance": metadata.get("performance", {}),
                    "use_cases": metadata.get("use_cases", []),
                    "files": {
                        "config": metadata.get("files", {}),  # Configuration from model_config.json
                        "discovered": model_info.get("files", [])  # Actual files found in folder
                    },
                    "model_type": metadata.get("model_type", "unknown"),
                    "size_mb": model_info.get("total_size_mb", 0),
                    "status": "available"
                })
            
            return {
                "success": True,
                "models": available_models,
                "categories": categories,
                "total_models": len(available_models)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "models": [],
                "categories": {},
                "total_models": 0
            }

    @web_app.post("/predict")
    async def predict_http(req: PredictRequest):
        """HTTP prediction endpoint for single images"""
        try:
            predictions = process_image.remote(req.image, req.width or 640, req.height or 480, req.model or "yolo")
            return {"success": True, "predictions": predictions}
        except Exception as e:
            return {"success": False, "error": str(e), "predictions": []}
    
    @web_app.post("/process_video")
    async def process_video_http(req: VideoProcessRequest):
        """HTTP endpoint for processing entire videos"""
        try:
            processed_frames = process_video.remote(req.video, req.frame_skip or 10, req.model or "yolo")
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
    print(f"GPU type: {GPU_TYPE}")
    print("Deploy with: modal deploy modal_service.py")
    print("For local dev with custom name: MODAL_APP_NAME=my-test-app modal deploy modal_service.py")
    print("To set GPU type: MODAL_GPU_TYPE=a100 modal deploy modal_service.py")
    print()
    print("Available commands:")
    print("- Process single image: modal run modal_service.py::process_image --image-b64 <base64_string>")
    print("- Start web server: modal serve modal_service.py")
    print()
    print("Model management (use manage_models.py):")
    print("- Download models: modal run manage_models.py::download_yolo_model")
    print("- List models: modal run manage_models.py::list_models")
    print("- Remove models: modal run manage_models.py::remove_model --model-name yolo")
