from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.config import settings


class FaceCompareRequest(BaseModel):
    image1: str = Field(..., description="First image (URL or Base64)")
    image2: str = Field(..., description="Second image (URL or Base64)")
    threshold: float = Field(settings.DEFAULT_THRESHOLD, description="Similarity threshold (0.0-1.0)")
    limit_faces: int = Field(settings.DEFAULT_LIMIT_FACES, description="Max faces per image (0=unlimited)")
    min_face_size: int = Field(settings.DEFAULT_MIN_FACE_SIZE, description="Minimum face size in pixels")
    return_face_data: bool = Field(settings.DEFAULT_RETURN_FACE_DATA, description="Return cropped face data")
    return_landmarks: bool = Field(settings.DEFAULT_RETURN_LANDMARKS, description="Return facial landmarks")
    detection_threshold: float = Field(settings.DEFAULT_DETECTION_THRESHOLD, description="Face detection confidence threshold")
    best_face_strategy: str = Field(settings.DEFAULT_BEST_FACE_STRATEGY, description="Strategy: 'area', 'center', 'confidence'")
    input_format: str = Field(settings.DEFAULT_INPUT_FORMAT, description="Input format hint")
    compare_all_faces: bool = Field(settings.DEFAULT_COMPARE_ALL_FACES, description="Compare all faces vs all faces")
    enable_rotation: bool = Field(settings.DEFAULT_ENABLE_ROTATION, description="Enable rotation logic if no match found")


class FaceCounts(BaseModel):
    image1: int
    image2: int

class CompareParameters(BaseModel):
    threshold: float
    detection_threshold: float
    limit_faces: int
    min_face_size: int
    best_face_strategy: str
    input_format: str
    enable_rotation: bool

class FaceCompareResponse(BaseModel):
    is_same_person: bool
    similarity_score: float
    status: str
    error_message: Optional[str] = None
    face_counts: FaceCounts
    processing_time_ms: float
    parameters: CompareParameters

