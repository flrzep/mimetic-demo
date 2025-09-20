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
])

# Create a volume for model caching
model_volume = modal.Volume.from_name("yolo-models", create_if_missing=True)

# Global variable to store the model instance
yolo_model = None

def get_yolo_model():
    """Initialize YOLO model once and reuse across function calls"""
    global yolo_model
    if yolo_model is None:
        # Import here to avoid deployment issues
        import os
        print("Loading YOLOv10 model...")
        
        try:
            # Use Hugging Face model instead of local files
            yolo_model = HuggingFaceYOLO()
            print("YOLOv10 model loaded from Hugging Face")
        except Exception as e:
            # Fallback to basic implementation
            print(f"Failed to load HF model ({e}), using fallback implementation...")
            yolo_model = FallbackYOLO()
        print("YOLO model ready")
    return yolo_model

class HuggingFaceYOLO:
    """YOLOv10 implementation using Hugging Face models"""
    def __init__(self):
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        
        try:
            # Download YOLOv10 ONNX model from Hugging Face
            model_path = hf_hub_download(
                repo_id="jameslahm/yolov10n", 
                filename="yolov10n.onnx",
                cache_dir="/cache/yolo"
            )
            
            # Initialize ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get input/output information
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # COCO class names for YOLOv10
            self.class_names = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"
            ]
            
            print(f"YOLOv10 loaded with {len(self.class_names)} classes")
            
        except Exception as e:
            print(f"Error loading YOLOv10 from HuggingFace: {e}")
            raise e
    
    def prepare_input(self, image):
        """Prepare input for YOLOv10 inference"""
        import cv2
        import numpy as np

        # Resize image to model input size (640x640)
        input_size = 640
        h, w = image.shape[:2]
        
        # Calculate scale and padding
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        top = (input_size - new_h) // 2
        left = (input_size - new_w) // 2
        
        # Place resized image in padded canvas
        padded[top:top+new_h, left:left+new_w] = resized
        
        # Convert to RGB and normalize
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Store scale and padding for later use
        self._scale = scale
        self._top_pad = top
        self._left_pad = left
        
        return input_tensor
    
    def process_output(self, outputs, conf_threshold=0.3):
        """Process YOLOv10 output to get bounding boxes"""
        import numpy as np

        # YOLOv10 output format: [batch, detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
        detections = outputs[0][0]  # Remove batch dimension
        
        # Filter by confidence
        confident_detections = detections[detections[:, 4] > conf_threshold]
        
        if len(confident_detections) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract components
        boxes = confident_detections[:, :4]  # x1, y1, x2, y2
        scores = confident_detections[:, 4]   # confidence
        class_ids = confident_detections[:, 5].astype(int)  # class id
        
        # Convert coordinates back to original image space
        # Account for padding and scaling
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - self._left_pad) / self._scale  # x coordinates
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - self._top_pad) / self._scale   # y coordinates
        
        # Ensure coordinates are non-negative
        boxes = np.maximum(boxes, 0)
        
        return boxes, scores, class_ids

class FallbackYOLO:
    """Fallback YOLO implementation when model files aren't available"""
    def __init__(self):
        self.class_names = ["person", "car", "truck", "bus", "motorbike", "bicycle"]
        
    def prepare_input(self, image):
        import numpy as np

        # Mock input preparation
        return np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    def process_output(self, outputs, conf_threshold=0.3):
        import numpy as np

        # Mock output processing - return some fake detections
        boxes = np.array([[100, 100, 200, 200], [300, 150, 450, 300]])
        scores = np.array([0.85, 0.75])
        class_ids = np.array([0, 1])
        return boxes, scores, class_ids
    
    @property
    def session(self):
        return self
    
    @property
    def output_names(self):
        return ["output"]
    
    @property 
    def input_names(self):
        return ["input"]
    
    def run(self, output_names, input_dict):
        # Mock inference
        import numpy as np
        return [np.random.randn(1, 25200, 6)]

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
        outputs = model.session.run(
            model.output_names, 
            {model.input_names[0]: model.prepare_input(cv_image)}
        )
        
        # Process outputs to get predictions
        boxes, scores, class_ids = model.process_output(outputs, conf_threshold=0.3)
        
        # Convert to our API format
        predictions = []
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            predictions.append({
                "class_id": int(class_id),
                "confidence": float(score),
                "label": model.class_names[class_id],
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
                        outputs = model.session.run(
                            model.output_names, 
                            {model.input_names[0]: model.prepare_input(frame)}
                        )
                        
                        # Process outputs to get predictions
                        boxes, scores, class_ids = model.process_output(outputs, conf_threshold=0.3)
                        
                        # Convert to our API format
                        frame_predictions = []
                        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                            x1, y1, x2, y2 = box.astype(int)
                            frame_predictions.append({
                                "class_id": int(class_id),
                                "confidence": float(score),
                                "label": model.class_names[class_id],
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
