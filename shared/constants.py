# API Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp']
REQUEST_TIMEOUT = 60  # seconds

# Model Configuration
SUPPORTED_MODELS = [
    'resnet50',
    'efficientnet_b0', 
    'yolov5',
    'custom'
]

# Response Codes
class ResponseCodes:
    SUCCESS = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    NOT_FOUND = 404
    RATE_LIMITED = 429
    SERVER_ERROR = 500
