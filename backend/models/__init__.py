from typing import Dict, List, Optional

from pydantic import BaseModel


class VideoProcessingRequest(BaseModel):
    video_data: str  # base64 encoded video data
    filename: str  # original filename for extension detection
    frame_interval: int = 1  # process every nth frame
    max_frames: Optional[int] = None  # limit total frames processed
    output_format: str = "mp4"  # "mp4", "webm", "frames", "annotations"
    video_codec: str = "h264"  # "h264", "h265", "vp8", "vp9"
    audio_codec: str = "none"  # "none", "aac", "mp3"
    return_url: bool = False  # return URL instead of base64
    model: str = "yolo"  # model to use for processing

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class Keypoint(BaseModel):
    x: float
    y: float
    score: Optional[float] = None

class PredictionResult(BaseModel):
    class_id: int
    confidence: float
    label: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    # Optional list of keypoints when model outputs them
    keypoints: Optional[List[Keypoint]] = None

class PredictionResponse(BaseModel):
    success: bool
    predictions: Optional[List[PredictionResult]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class VideoFrame(BaseModel):
    frame_number: int
    timestamp: float
    predictions: List[PredictionResult]
    processed_frame: Optional[str] = None  # base64 encoded

class VideoProcessingResponse(BaseModel):
    success: bool
    total_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    processing_time: Optional[float] = None
    frames: Optional[List[VideoFrame]] = None
    output_video: Optional[str] = None  # base64 encoded processed video
    output_video_url: Optional[str] = None  # URL to processed video
    output_format: Optional[str] = None  # actual format used
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    backend_status: str
    modal_status: dict
    response_time: float
    timestamp: str
