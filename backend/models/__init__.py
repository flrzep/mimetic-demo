from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    image: str  # base64 encoded

class PredictionResult(BaseModel):
    class_id: int
    confidence: float
    label: Optional[str] = None

class PredictionResponse(BaseModel):
    success: bool
    predictions: Optional[List[PredictionResult]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    backend_status: str
    modal_status: dict
    response_time: float
    timestamp: str
