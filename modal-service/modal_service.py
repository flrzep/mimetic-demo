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
])

# Create the Modal app with branch-based naming
app = modal.App(app_name, image=image)

# Create a volume for model caching
model_volume = modal.Volume.from_name("yolo-models", create_if_missing=True)

# Global variable to store the model instance
yolo_model = None

class YOLOv10:
    """YOLOv10 implementation using ONNX runtime"""
    def __init__(self, cache_dir):
        import onnxruntime
        from huggingface_hub import hf_hub_download

        # Initialize model
        self.cache_dir = cache_dir
        print(f"Initializing YOLO model from {self.cache_dir}")
        model_file = hf_hub_download(
            repo_id="onnx-community/yolov10n",
            filename="onnx/model.onnx",
            cache_dir=self.cache_dir,
        )
        self.initialize_model(model_file)
        print("YOLO model initialized")

    def initialize_model(self, model_file):
        import numpy as np
        import onnxruntime
        
        self.session = onnxruntime.InferenceSession(
            model_file,
            providers=[
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(self.cache_dir) + "/onnx.cache",
                    },
                ),
                "CUDAExecutionProvider",
            ],
        )
        # Get model info
        self.get_input_details()
        self.get_output_details()

        # COCO class names
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(self.class_names), 3))

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_width = model_inputs[0].shape[2]
        self.input_height = model_inputs[0].shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_output.name for model_output in model_outputs]

    def preprocess(self, image):
        import cv2
        import numpy as np

        # Get image dimensions
        self.img_height, self.img_width = image.shape[:2]

        # Resize image to match model input
        input_img = cv2.resize(image, (self.input_width, self.input_height))

        # Normalize pixel values to range [0, 1]
        input_img = input_img / 255.0

        # Transpose to match PyTorch format (C, H, W)
        input_img = input_img.transpose(2, 0, 1)

        # Add batch dimension
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def postprocess(self, input_img, output):
        import numpy as np
        
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > 0.3, :]
        scores = scores[scores > 0.3]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
        indices = self.apply_nms(boxes, scores)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        import numpy as np

        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        import numpy as np

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def xywh2xyxy(self, x):
        # Convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
        y = x.copy()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def apply_nms(self, boxes, scores):
        import cv2
        import numpy as np

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.4)
        return indices.flatten() if len(indices) > 0 else np.array([])

    def predict(self, image):
        import numpy as np
        
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.session.get_inputs()[0].name: input_tensor})
        boxes, scores, class_ids = self.postprocess(input_tensor, outputs)

        # Convert results to list of dictionaries
        results = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            results.append({
                "class": self.class_names[class_ids[i]],
                "confidence": float(scores[i]),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        return results



def get_yolo_model():
    """Initialize YOLO model once and reuse across function calls"""
    global yolo_model
    if yolo_model is None:
        print("Loading YOLOv10 model...")
        
        import onnxruntime
        onnxruntime.preload_dlls()
        
        cache_path = "/cache/yolo"
        import os
        os.makedirs(cache_path, exist_ok=True)
        
        # Use the embedded YOLOv10 implementation
        yolo_model = YOLOv10(cache_path)
        print("YOLOv10 model loaded successfully")
        
        print("YOLO model ready")
    return yolo_model

@app.function(
    image=image,
    gpu="any",
    volumes={"/cache": model_volume},
    scaledown_window=300,
    timeout=3600
)
def process_image(image_b64: str, width: int = 640, height: int = 480) -> List[Dict]:
    '''
    Run YOLOv10 inference on a single image
    '''

    try:
        # Import inside function to avoid Modal deployment issues
        import cv2
        import numpy as np
        from PIL import Image

        print(f"Processing image with YOLOv10: {width}x{height}")

        # Get the shared YOLO model instance
        model = get_yolo_model()

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
    volumes={"/cache": model_volume},
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

        import cv2
        import numpy as np

        # Set environment variables for headless OpenCV
        os.environ['DISPLAY'] = ''
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        print(f"Starting video processing with YOLOv10, frame_skip={frame_skip}")
        
        # Get the shared YOLO model instance
        model = get_yolo_model()
        
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
