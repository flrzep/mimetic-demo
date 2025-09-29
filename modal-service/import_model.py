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
import torch
import torchvision.transforms as transforms

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
    """YOLOv10 object detection model supporting both PyTorch and ONNX runtime."""

    def __init__(self, model_folder="yolo", weights_file=None, classes_file=None, model_type="pytorch"):
        self.cache_dir = Path("/cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model runtime instances (only one will be used)
        self.session = None  # ONNX runtime session
        self.torch_model = None  # PyTorch model
        
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
            # Download from HuggingFace as fallback (prefer ONNX for fallback)
            print("Downloading YOLOv10 model from HuggingFace...")
            from huggingface_hub import hf_hub_download
            
            model_file = hf_hub_download(
                repo_id="onnx-community/yolov10n",
                filename="onnx/model.onnx",
                local_dir="/tmp",
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded model to {model_file}")
            # Override model_type for downloaded ONNX model
            self.model_type = "onnx"
        
        # Determine actual model type from file extension if not explicitly set
        if model_file.endswith(('.onnx',)):
            actual_model_type = "onnx"
        elif model_file.endswith(('.pt', '.pth')):
            actual_model_type = "pytorch"
        else:
            # Use configured model_type as fallback
            actual_model_type = self.model_type
            
        print(f"Loading model: {model_file} (type: {actual_model_type})")
        self.initialize_model(model_file, actual_model_type)

    def initialize_model(self, model_file, model_type):
        """Initialize either PyTorch or ONNX model based on type."""
        self.actual_model_type = model_type
        
        if model_type == "onnx":
            print("Initializing ONNX model...")
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
                    "CPUExecutionProvider",  # Fallback
                ],
            )
            # Get model info for ONNX
            self.get_input_details()
            self.get_output_details()
            
        elif model_type == "pytorch":
            print("Initializing PyTorch model...")
            # Load PyTorch model
            self.torch_model = torch.jit.load(model_file, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            self.torch_model.eval()
            
            # Set standard YOLO input dimensions (will be validated during first inference)
            self.input_width = 640
            self.input_height = 640
            self.img_width = 640
            self.img_height = 640
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        print(f"Model initialized successfully with {model_type} runtime")

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
        if self.actual_model_type == "onnx":
            return self._predict_onnx(image, conf_threshold)
        elif self.actual_model_type == "pytorch":
            return self._predict_pytorch(image, conf_threshold)
        else:
            raise ValueError(f"Unsupported model type: {self.actual_model_type}")
    
    def _predict_onnx(self, image, conf_threshold=0.3):
        """ONNX-specific prediction logic."""
        input_tensor = self.prepare_input(image)
        
        # Run inference and get raw results
        onnxruntime.set_seed(42)
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )
        
        # Process outputs to get boxes, scores, class_ids
        boxes, scores, class_ids = self.process_output(outputs, conf_threshold)
        
        return self._format_results(boxes, scores, class_ids)
    
    def _predict_pytorch(self, image, conf_threshold=0.3):
        """PyTorch-specific prediction logic."""
        input_tensor = self.prepare_input_pytorch(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.torch_model(input_tensor)
        
        # Convert to numpy for processing
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs.cpu().numpy()]
        elif isinstance(outputs, (list, tuple)):
            outputs = [output.cpu().numpy() for output in outputs]
        
        # Process outputs to get boxes, scores, class_ids
        boxes, scores, class_ids = self.process_output(outputs, conf_threshold)
        
        return self._format_results(boxes, scores, class_ids)
    
    def _format_results(self, boxes, scores, class_ids):
        """Convert predictions to API format."""
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
    
    def prepare_input_pytorch(self, image):
        """Prepare input tensor for PyTorch model."""
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(input_img[np.newaxis, :, :, :]).float()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
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
        if hasattr(self, 'actual_model_type') and self.actual_model_type == "onnx":
            model_inputs = self.session.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

            self.input_shape = model_inputs[0].shape
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
        # For PyTorch models, input details are set in initialize_model

    def get_output_details(self):
        if hasattr(self, 'actual_model_type') and self.actual_model_type == "onnx":
            model_outputs = self.session.get_outputs()
            self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        # For PyTorch models, output details are handled dynamically

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


class KeypointsONNX:
    """Generic ONNX keypoint detection wrapper.

    Tries to support common output conventions:
    - Keypoint R-CNN style: boxes (N,4), labels (N,), scores (N,), keypoints (N,K,2 or N,K,3)
    - YOLO pose-style: predictions (N, 6 + K*3) with [x1,y1,x2,y2,conf,class_id,kp1x,kp1y,kp1s,...]

    Returns results in the same API format used by YOLOv10.predict():
    [ { "class": str, "confidence": float, "bbox": [x1,y1,x2,y2], "keypoints"?: [ {x,y,score?}, ... ] } ]
    """

    def __init__(self, model_folder: str, weights_file: str, classes_file: str = None):
        self.model_folder = model_folder
        self.weights_file = weights_file
        self.classes_file = classes_file

        self.session = None
        self.input_names = []
        self.output_names = []
        self.input_height = None  # type: ignore
        self.input_width = None   # type: ignore
        self.img_height = None
        self.img_width = None

        self.class_names = None  # type: ignore
        self.keypoint_names = None  # type: ignore

        # Optional preprocessing config
        self.mean = None  # type: ignore
        self.std = None   # type: ignore
        self.scale = 1.0

        self._load_classes()
        self._load_model()

    def _load_classes(self):
        # Try YAML first (may include keypoint names)
        import json
        import yaml
        self.class_names = None
        self.keypoint_names = None
        if self.classes_file:
            yaml_path = Path(f"/models/{self.model_folder}/{self.classes_file}")
            if yaml_path.suffix.lower() in (".yml", ".yaml") and yaml_path.exists():
                try:
                    with open(yaml_path, "r") as f:
                        data = yaml.safe_load(f) or {}
                        self.class_names = data.get("classes") or data.get("labels")
                        self.keypoint_names = data.get("keypoints") or data.get("kp")
                        if self.class_names:
                            self.class_names = [str(c) for c in self.class_names]
                        if self.keypoint_names:
                            self.keypoint_names = [str(k) for k in self.keypoint_names]
                        print(f"Loaded YAML classes/keypoints from {yaml_path}")
                        # Optional preprocessing: { mean: [..], std: [..], scale: 255 or 1.0 }
                        pp = data.get("preprocess") or {}
                        if isinstance(pp, dict):
                            m = pp.get("mean")
                            s = pp.get("std")
                            sc = pp.get("scale")
                            try:
                                if isinstance(m, (list, tuple)) and len(m) == 3:
                                    self.mean = np.array(m, dtype=np.float32)
                                if isinstance(s, (list, tuple)) and len(s) == 3:
                                    self.std = np.array(s, dtype=np.float32)
                                if isinstance(sc, (int, float)):
                                    self.scale = float(sc)
                            except Exception as e:
                                print(f"Warning: invalid preprocess config in YAML: {e}")
                except Exception as e:
                    print(f"Warning: failed parsing YAML classes file {yaml_path}: {e}")

        # Fallback to txt classes
        if self.class_names is None:
            fallback_txts = [
                Path(f"/models/{self.model_folder}/classes.txt"),
                Path(f"/models/{self.model_folder}/yolo_classes.txt"),
                Path("/models/yolo/yolo_classes.txt"),
                this_dir / "models" / "yolo" / "yolo_classes.txt",
            ]
            for p in fallback_txts:
                try:
                    if p.exists():
                        with open(p, "r") as f:
                            self.class_names = [line.strip() for line in f if line.strip()]
                            print(f"Loaded class names from {p}")
                            break
                except Exception:
                    continue

        if self.class_names is None:
            print("No classes file found; defaulting to ['object']")
            self.class_names = ["object"]

    def _load_model(self):
        model_path = f"/models/{self.model_folder}/{self.weights_file}"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Keypoints ONNX model not found at {model_path}")

        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "/cache/onnx.cache",
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        print(f"Initializing ONNXRuntime session for keypoints model: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self._get_io_details()

    def _get_io_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [i.name for i in model_inputs]
        shape = model_inputs[0].shape  # [N,C,H,W]
        # Some models export dynamic dims like 'None' or string names
        try:
            self.input_height = int(shape[2]) if isinstance(shape[2], (int, np.integer)) else None
            self.input_width = int(shape[3]) if isinstance(shape[3], (int, np.integer)) else None
        except Exception:
            self.input_height = None
            self.input_width = None

        model_outputs = self.session.get_outputs()
        self.output_names = [o.name for o in model_outputs]
        print(f"Model IO: inputs={self.input_names}, outputs={self.output_names}, size={self.input_width}x{self.input_height}")

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Use dynamic input if model allows; otherwise resize to fixed size
        if self.input_width is not None and self.input_height is not None:
            if (img.shape[1], img.shape[0]) != (self.input_width, self.input_height):
                img = cv2.resize(img, (self.input_width, self.input_height))
        else:
            # No fixed input size -> keep original and remember for scaling
            self.input_width = img.shape[1]
            self.input_height = img.shape[0]

        # Scale/normalize
        img = img.astype(np.float32)
        if self.scale and self.scale != 1.0:
            img = img / self.scale
        else:
            img = img / 255.0
        if self.mean is not None and self.std is not None:
            # Expect mean/std in RGB order; img is HWC in 0..1
            img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # HWC->CHW
        return np.expand_dims(img, axis=0)

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25):
        inp = self.prepare_input(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: inp})

        # Try parsing as Keypoint R-CNN first
        parsed = self._parse_keypoint_rcnn(outputs, conf_threshold)
        if parsed is None:
            parsed = self._parse_yolo_pose(outputs, conf_threshold)
        if parsed is None:
            print("Warning: could not parse ONNX keypoint outputs; returning empty results")
            return []
        return parsed

    def _parse_keypoint_rcnn(self, outputs, conf_threshold):
        # Expect arrays among outputs: boxes(N,4), labels(N), scores(N), keypoints(N,K,2or3)
        boxes = labels = scores = keypoints = None
        for arr in outputs:
            if not isinstance(arr, np.ndarray):
                continue
            if arr.ndim == 2 and arr.shape[1] == 4 and boxes is None:
                boxes = arr
            elif arr.ndim == 1 and arr.dtype.kind in ("i", "u") and labels is None:
                labels = arr
            elif arr.ndim == 1 and arr.dtype.kind == "f" and scores is None:
                scores = arr
            elif arr.ndim == 3 and keypoints is None:
                keypoints = arr

        if boxes is None or labels is None or scores is None:
            return None

        n = min(len(boxes), len(labels), len(scores))
        if keypoints is not None:
            n = min(n, keypoints.shape[0])
        if n == 0:
            return []

        results = []
        sx = self.img_width / self.input_width
        sy = self.img_height / self.input_height
        for i in range(n):
            conf = float(scores[i])
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = boxes[i]
            # scale to original image size if inputs are in model scale
            x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            label_id = int(labels[i])
            label = self.class_names[label_id] if 0 <= label_id < len(self.class_names) else str(label_id)
            pred = {
                "class": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
            if keypoints is not None and i < keypoints.shape[0]:
                kps = keypoints[i]
                kps_out = []
                # kps shape: (K,2) or (K,3)
                for kp in kps:
                    if kp.shape[-1] >= 2:
                        xk = float(kp[0]) * sx
                        yk = float(kp[1]) * sy
                        item = {"x": xk, "y": yk}
                        if kp.shape[-1] >= 3:
                            # Some models output visibility {0,1,2}; treat as score but keep value
                            item["score"] = float(kp[2])
                        kps_out.append(item)
                if kps_out:
                    pred["keypoints"] = kps_out
            results.append(pred)
        return results

    def _parse_yolo_pose(self, outputs, conf_threshold):
        # Assume single output [N, 6 + K*3] -> [x1,y1,x2,y2,conf,cls,kps...]
        if not outputs:
            return None
        out = outputs[0]
        if not isinstance(out, np.ndarray):
            return None
        pred = np.squeeze(out)
        if pred.ndim != 2 or pred.shape[1] < 6:
            return None

        n, d = pred.shape
        rest = d - 6
        K = rest // 3 if rest >= 3 else 0
        results = []
        sx = self.img_width / self.input_width
        sy = self.img_height / self.input_height
        for i in range(n):
            x1, y1, x2, y2, conf, cls_id = pred[i, :6]
            conf = float(conf)
            if conf < conf_threshold:
                continue
            label_id = int(cls_id)
            label = self.class_names[label_id] if 0 <= label_id < len(self.class_names) else str(label_id)
            res = {
                "class": label,
                "confidence": conf,
                "bbox": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
            }
            if K > 0:
                kps = pred[i, 6:6 + 3 * K].reshape(K, 3)
                kp_list = []
                for (xk, yk, sk) in kps:
                    kp_list.append({"x": float(xk * sx), "y": float(yk * sy), "score": float(sk)})
                res["keypoints"] = kp_list
            results.append(res)

        return results
