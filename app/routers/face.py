from fastapi import APIRouter, HTTPException, Body
from app.services.face_service import face_service
from app.schemas import FaceCompareRequest, FaceCompareResponse

router = APIRouter(prefix="/face", tags=["face"])

@router.post("/compare", response_model=FaceCompareResponse)
async def compare_faces(request: FaceCompareRequest):
    try:
        # Pass request fields as keyword arguments to the service
        # We can dump the model to dict, but need to be careful with keys matching arguments
        result = face_service.compare_faces(**request.model_dump())
        
        return result
    except Exception as e:
        # In case of unexpected server error not handled by service
        raise HTTPException(status_code=500, detail=str(e))
