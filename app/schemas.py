from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from app.config import settings


FaceStrategy = Literal["area", "center", "confidence", "score"]


class FaceCompareRequest(BaseModel):
    image1: str = Field(..., description="First image (URL or Base64)")
    image2: str = Field(..., description="Second image (URL or Base64)")
    threshold: float = Field(
        settings.DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0.0-1.0)",
    )
    limit_faces: int = Field(
        settings.DEFAULT_LIMIT_FACES,
        ge=0,
        description="Max faces per image (0=unlimited)",
    )
    min_face_size: int = Field(
        settings.DEFAULT_MIN_FACE_SIZE,
        ge=0,
        description="Minimum face size in pixels",
    )
    detection_threshold: float = Field(
        settings.DEFAULT_DETECTION_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Face detection confidence threshold",
    )
    best_face_strategy: FaceStrategy = Field(
        settings.DEFAULT_BEST_FACE_STRATEGY,
        description="Strategy: 'area', 'center', 'confidence', 'score'",
    )
    compare_all_faces: bool = Field(
        settings.DEFAULT_COMPARE_ALL_FACES,
        description="Compare all faces vs all faces",
    )
    enable_rotation: bool = Field(
        settings.DEFAULT_ENABLE_ROTATION,
        description="Enable rotation logic if no match found",
    )


class FaceCounts(BaseModel):
    image1: int
    image2: int


class CompareParameters(BaseModel):
    threshold: float
    detection_threshold: float
    limit_faces: int
    min_face_size: int
    best_face_strategy: FaceStrategy
    enable_rotation: bool


class FaceCompareResult(BaseModel):
    is_same_person: bool
    similarity_score: float
    face_counts: FaceCounts
    processing_time_ms: float
    parameters: CompareParameters


T = TypeVar("T")


class StandardResponse(BaseModel, Generic[T]):
    code: int = Field(..., description="Status code (200 for success)")
    msg: str = Field(..., description="Message")
    data: Optional[T] = Field(None, description="Data payload")


class FaceCompareResponse(StandardResponse[FaceCompareResult]):
    pass
