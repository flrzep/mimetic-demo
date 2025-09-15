import base64
import io
import logging
import time
from typing import Any, Optional, Tuple

from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv", "video/webm"}


def validate_image_file(filename: str, content_type: str, max_size: int, size_bytes: int) -> Tuple[bool, str]:
    logger.info(f"Validating image file: {filename}, type: {content_type}, size: {size_bytes} bytes")
    
    if content_type not in ALLOWED_TYPES:
        logger.warning(f"Invalid image type: {content_type} for file {filename}")
        return False, "Invalid file type. Only PNG, JPG, JPEG, WEBP allowed."
    if size_bytes > max_size:
        logger.warning(f"Image file too large: {size_bytes} bytes (max: {max_size}) for file {filename}")
        return False, "File too large. Max 10MB allowed."
    if not filename:
        logger.warning("Image filename missing")
        return False, "Filename missing."
    
    logger.info(f"Image validation successful for {filename}")
    return True, ""


def validate_video_file(filename: str, content_type: str, max_size: int, size_bytes: int) -> Tuple[bool, str]:
    """Validate video file upload"""
    logger.info(f"Validating video file: {filename}, type: {content_type}, size: {size_bytes} bytes")
    
    if content_type not in ALLOWED_VIDEO_TYPES:
        logger.warning(f"Invalid video type: {content_type} for file {filename}")
        return False, "Invalid file type. Only MP4, AVI, MOV, MKV, WEBM allowed."
    if size_bytes > max_size:
        logger.warning(f"Video file too large: {size_bytes} bytes (max: {max_size}) for file {filename}")
        return False, f"File too large. Max {max_size // (1024 * 1024)}MB allowed."
    if not filename:
        logger.warning("Video filename missing")
        return False, "Filename missing."
    
    logger.info(f"Video validation successful for {filename}")
    return True, ""


def convert_to_base64(file_content: bytes) -> str:
    return base64.b64encode(file_content).decode("utf-8")


def resize_image_if_needed(image_data: bytes, max_size: int = 1024) -> bytes:
    logger.info(f"Checking if image resize needed (max_size: {max_size})")
    
    with Image.open(io.BytesIO(image_data)) as img:
        img_format = img.format
        w, h = img.size
        original_size = len(image_data)
        
        logger.info(f"Original image: {w}x{h}, format: {img_format}, size: {original_size} bytes")
        
        if max(w, h) <= max_size:
            logger.info("Image resize not needed")
            return image_data
            
        ratio = max_size / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        
        logger.info(f"Resizing image from {w}x{h} to {new_size[0]}x{new_size[1]} (ratio: {ratio:.3f})")
        
        img = img.resize(new_size, Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format=img_format or 'PNG', quality=90)
        resized_data = out.getvalue()
        
        logger.info(f"Image resized successfully. New size: {len(resized_data)} bytes ({len(resized_data)/original_size:.2%} of original)")
        return resized_data


def log_request(endpoint: str, processing_time: float, success: bool, additional_info: Optional[dict] = None):
    """Enhanced logging for API requests"""
    status = "SUCCESS" if success else "ERROR"
    log_msg = f"[API] {endpoint} - {status} - {processing_time:.3f}s"
    
    if additional_info:
        info_str = ", ".join([f"{k}: {v}" for k, v in additional_info.items()])
        log_msg += f" - {info_str}"
    
    if success:
        logger.info(log_msg)
    else:
        logger.error(log_msg)


def log_processing_step(step: str, details: Optional[dict] = None):
    """Log individual processing steps"""
    log_msg = f"[PROCESSING] {step}"
    if details:
        info_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
        log_msg += f" - {info_str}"
    logger.info(log_msg)


def log_video_processing(event: str, frame_number: Optional[int] = None, **kwargs):
    """Specialized logging for video processing"""
    log_msg = f"[VIDEO] {event}"
    if frame_number is not None:
        log_msg += f" [Frame {frame_number}]"
    if kwargs:
        info_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
        log_msg += f" - {info_str}"
    logger.info(log_msg)
