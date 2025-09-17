# Modal GPU service for computer vision inference
import base64
import io
import os
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
def predict_image(image_b64: str, width: int = 640, height: int = 480) -> List[Dict]:
    """
    Run inference on a single image
    Replace this with your actual model inference
    """
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

# Create FastAPI app inside function to avoid import issues
def create_web_app():
    from fastapi import FastAPI
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
    
    web_app = FastAPI(title="Modal CV Service")

    @web_app.get("/health")
    async def health_http():
        """HTTP health check"""
        return {"status": "ok", "model": "yolo", "gpu": True}

    @web_app.post("/predict")
    async def predict_http(req: PredictRequest):
        """HTTP prediction endpoint"""
        try:
            predictions = predict_image.remote(req.image, req.width or 640, req.height or 480)
            return {"success": True, "predictions": predictions}
        except Exception as e:
            return {"success": False, "error": str(e), "predictions": []}
    
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
