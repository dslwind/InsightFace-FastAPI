from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class FaceCompareRequest(BaseModel):
    image1: str = Field(..., description="First image (URL or Base64)")
    image2: str = Field(..., description="Second image (URL or Base64)")
    threshold: float = Field(0.3, description="Similarity threshold (0.0-1.0)")
    limit_faces: int = Field(0, description="Max faces per image (0=unlimited)")
    min_face_size: int = Field(0, description="Minimum face size in pixels")
    return_face_data: bool = Field(False, description="Return cropped face data")
    return_landmarks: bool = Field(False, description="Return facial landmarks")
    detection_threshold: float = Field(0.6, description="Face detection confidence threshold")
    best_face_strategy: str = Field("center", description="Strategy: 'area', 'center', 'confidence'")
    input_format: str = Field("auto", description="Input format hint")
    compare_all_faces: bool = Field(False, description="Compare all faces vs all faces")

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

class FaceCompareResponse(BaseModel):
    is_same_person: bool
    similarity_score: float
    status: str
    error_message: Optional[str] = None
    face_counts: FaceCounts
    processing_time_ms: float
    parameters: CompareParameters

