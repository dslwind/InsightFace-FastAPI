from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.face_service import face_service
from app.schemas import FaceCompareResponse

router = APIRouter(prefix="/face", tags=["face"])

@router.post("/compare", response_model=FaceCompareResponse)
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Basic validation of content type if provided
    # Note: content_type is trusted from client, but actual decoding happens in cv2
    valid_types = ["image/jpeg", "image/png", "image/jpg"]
    # Check only if content_type is available, sometimes it might be generic application/octet-stream
    if file1.content_type and file1.content_type not in valid_types and file1.content_type != 'application/octet-stream':
         raise HTTPException(status_code=400, detail="Invalid file type for file1. Only JPEG/PNG allowed.")
    if file2.content_type and file2.content_type not in valid_types and file2.content_type != 'application/octet-stream':
         raise HTTPException(status_code=400, detail="Invalid file type for file2. Only JPEG/PNG allowed.")
    
    try:
        img1_bytes = await file1.read()
        img2_bytes = await file2.read()
        
        result = face_service.compare_faces(img1_bytes, img2_bytes)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
