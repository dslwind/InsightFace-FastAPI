from pydantic import BaseModel
from typing import Optional

class FaceCompareResponse(BaseModel):
    similarity: float
    match: bool
    error: Optional[str] = None
