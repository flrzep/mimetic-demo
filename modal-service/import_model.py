"""
Generic model loader for any model type in models/ directory.
Dynamically imports and initializes model classes based on model name.
"""

import importlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

this_dir = Path(__file__).parent

def get_model_class(model_name="yolo"):
    """
    Dynamically import and return the model class from models/{model_name}/{model_name}.py
    
    Args:
        model_name: Name of the model directory and Python file (e.g., "yolo", "sam", "clip")
    
    Returns:
        The model class from the specified module
    """
    try:
        # Construct module path: models.{model_name}.{model_name}
        module_path = f"models.{model_name}.{model_name}"
        
        # Add the models directory to Python path if not already there
        models_dir = this_dir / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))
        
        # Import the module
        model_module = importlib.import_module(module_path)
        
        # Get the model class (assuming class name is capitalized model_name + "v10" for YOLO, or just capitalized for others)
        if model_name.lower() == "yolo":
            model_class = getattr(model_module, "YOLOv10")
        else:
            # For other models, try common naming patterns
            possible_names = [
                model_name.upper(),  # ALL_CAPS
                model_name.capitalize(),  # Capitalized
                f"{model_name.capitalize()}Model",  # With "Model" suffix
                f"{model_name.upper()}Model"  # ALL_CAPS with "Model" suffix
            ]
            
            model_class = None
            for name in possible_names:
                if hasattr(model_module, name):
                    model_class = getattr(model_module, name)
                    break
            
            if model_class is None:
                raise AttributeError(f"Could not find model class in {module_path}. Tried: {possible_names}")
        
        return model_class
        
    except ImportError as e:
        raise ImportError(f"Could not import model from {module_path}: {e}")


class YOLOv10:
    """YOLOv10 object detection model with ONNX runtime."""

    def __init__(self, model_folder="yolo", weights_file=None, classes_file=None, model_type="pytorch"):
        self.cache_dir = Path("/cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.session = None
        self.class_names = None
        self.colors = None
        
        # Model configuration from config file
        self.model_folder = model_folder
        self.weights_file = weights_file
        self.classes_file = classes_file
        self.model_type = model_type
        
        # Model dimensions - will be set when model is loaded
        self.input_height = None
        self.input_width = None
        self.img_height = None
        self.img_width = None
        
        # Model I/O details
        self.input_names = None
        self.output_names = None
        
        # Load the model
        self.load_model()
        print(f"YOLO model initialized from {model_folder} folder")

    def load_model(self):
        """Load the YOLO model from configured paths or fallback locations."""
        # Build model paths based on configuration
        model_paths = []
        
        if self.weights_file:
            # Use configured path
            configured_path = f"/models/{self.model_folder}/{self.weights_file}"
            model_paths.append(configured_path)
            print(f"Using configured model path: {configured_path}")
        
        # Fallback paths for backward compatibility
        model_paths.extend([
            f"/models/{self.model_folder}/yolov10n.onnx",
            f"/models/{self.model_folder}/yolov10b.pt", 
            "/models/yolo/yolov10n.onnx",  # Original hardcoded path
            "/tmp/yolov10n.onnx",  # Download fallback
        ])
        
        model_file = None
        for path in model_paths:
            print(f"Checking model path: {path}")
            if os.path.exists(path):
                model_file = path
                print(f"Using model from {path}")
                break
        
        if model_file is None:
            # Download from HuggingFace as fallback
            print("Downloading YOLOv10 model from HuggingFace...")
            from huggingface_hub import hf_hub_download
            
            model_file = hf_hub_download(
                repo_id="onnx-community/yolov10n",
                filename="onnx/model.onnx",
                local_dir="/tmp",
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded model to {model_file}")
        
        self.initialize_model(model_file)

    def initialize_model(self, model_file):
        self.session = onnxruntime.InferenceSession(
            model_file,
            providers=[
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": self.cache_dir / "onnx.cache",
                    },
                ),
                "CUDAExecutionProvider",
            ],
        )
        # Get model info
        self.get_input_details()
        self.get_output_details()

        # Load class names from configured and fallback paths
        classes_file_paths = []
        
        if self.classes_file:
            # Use configured classes file
            configured_classes_path = f"/models/{self.model_folder}/{self.classes_file}"
            classes_file_paths.append(configured_classes_path)
            print(f"Using configured classes file: {configured_classes_path}")
        
        # Fallback paths for backward compatibility
        classes_file_paths.extend([
            f"/models/{self.model_folder}/yolo_classes.txt",  # Folder-specific fallback
            "/models/yolo/yolo_classes.txt",  # Original hardcoded path
            this_dir / "models" / "yolo" / "yolo_classes.txt",    # Local fallback
        ])
        
        self.class_names = None
        for classes_file in classes_file_paths:
            try:
                with open(classes_file, "r") as f:
                    self.class_names = f.read().splitlines()
                    print(f"Loaded class names from {classes_file}")
                    break
            except (FileNotFoundError, OSError):
                continue
        
        if self.class_names is None:
            raise FileNotFoundError("Could not find yolo_classes.txt in any expected location")
            
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(self.class_names), 3))

    def detect_objects(self, image, conf_threshold=0.3):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        new_image = self.inference(image, input_tensor, conf_threshold)

        return new_image

    def predict(self, image, conf_threshold=0.3):
        """
        Predict objects in image and return results in API format
        Compatible with the Modal service API expectations
        """
        input_tensor = self.prepare_input(image)
        
        # Run inference and get raw results
        onnxruntime.set_seed(42)
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )
        
        # Process outputs to get boxes, scores, class_ids
        boxes, scores, class_ids = self.process_output(outputs, conf_threshold)
        
        # Convert to API format
        results = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            results.append({
                "class": self.class_names[class_ids[i]],
                "confidence": float(scores[i]),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        
        return results

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, image, input_tensor, conf_threshold=0.3):
        # set seed to potentially create smoother output in RT setting
        onnxruntime.set_seed(42)
        # start = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

        # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        (
            boxes,
            scores,
            class_ids,
        ) = self.process_output(outputs, conf_threshold)
        return self.draw_detections(image, boxes, scores, class_ids)

    def process_output(self, output, conf_threshold=0.3):
        predictions = np.squeeze(output[0])

        # Filter out object confidence scores below threshold
        scores = predictions[:, 4]
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = predictions[:, 5].astype(int)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        # boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [
                self.input_width,
                self.input_height,
                self.input_width,
                self.input_height,
            ]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes

    def draw_detections(
        self, image, boxes, scores, class_ids, draw_scores=True, mask_alpha=0.4
    ):
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0012
        text_thickness = int(min([img_height, img_width]) * 0.004)

        # Draw bounding boxes and labels of detections
        for class_id, box, score in zip(class_ids, boxes, scores):
            color = self.colors[class_id]

            self.draw_box(det_img, box, color)  # type: ignore

            label = self.class_names[class_id]
            caption = f"{label} {int(score * 100)}%"
            self.draw_text(det_img, caption, box, color, font_size, text_thickness)  # type: ignore

        return det_img

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def draw_box(
        self,
        image: np.ndarray,
        box: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 5,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    def draw_text(
        self,
        image: np.ndarray,
        text: str,
        box: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        font_size: float = 0.100,
        text_thickness: int = 5,
        box_thickness: int = 5,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        (tw, th), _ = cv2.getTextSize(
            text=text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_size,
            thickness=text_thickness,
        )
        x1 = x1 - box_thickness
        th = int(th * 1.2)

        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        return cv2.putText(
            image,
            text,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
