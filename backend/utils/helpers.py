import io
import time
import base64
from typing import Tuple
from PIL import Image

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def validate_image_file(filename: str, content_type: str, max_size: int, size_bytes: int) -> Tuple[bool, str]:
    if content_type not in ALLOWED_TYPES:
        return False, "Invalid file type. Only PNG, JPG, JPEG, WEBP allowed."
    if size_bytes > max_size:
        return False, "File too large. Max 10MB allowed."
    if not filename:
        return False, "Filename missing."
    return True, ""


def convert_to_base64(file_content: bytes) -> str:
    return base64.b64encode(file_content).decode("utf-8")


def resize_image_if_needed(image_data: bytes, max_size: int = 1024) -> bytes:
    with Image.open(io.BytesIO(image_data)) as img:
        img_format = img.format
        w, h = img.size
        if max(w, h) <= max_size:
            return image_data
        ratio = max_size / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format=img_format or 'PNG', quality=90)
        return out.getvalue()


def log_request(endpoint: str, processing_time: float, success: bool):
    # Basic log to stdout; integrate with real logging in production
    status = "success" if success else "error"
    print(f"[api] {endpoint} {status} {processing_time:.2f}s")
