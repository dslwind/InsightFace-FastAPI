from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional
from app.services.face_service import face_service
from app.schemas import FaceCompareResponse

router = APIRouter(prefix="/face", tags=["face"])

@router.post("/compare", response_model=FaceCompareResponse)
async def compare_faces(
    file1: Optional[UploadFile] = File(None),
    image1: Optional[str] = Form(None),
    file2: Optional[UploadFile] = File(None),
    image2: Optional[str] = Form(None),
    strategy: str = Form("largest", description="Face selection strategy: 'largest', 'center', 'score'")
):
    # Basic validation helper
    def validate_file(f: UploadFile):
        valid_types = ["image/jpeg", "image/png", "image/jpg"]
        # Check only if content_type is available, sometimes it might be generic application/octet-stream
        if f.content_type and f.content_type not in valid_types and f.content_type != 'application/octet-stream':
             return False
        return True

    # Get content for Image 1
    input1 = None
    if file1:
        if not validate_file(file1):
            raise HTTPException(status_code=400, detail="Invalid file type for file1. Only JPEG/PNG allowed.")
        input1 = await file1.read()
    elif image1:
        input1 = image1
    else:
        raise HTTPException(status_code=400, detail="Image 1 is missing. Provide 'file1' or 'image1' (url/path/base64).")

    # Get content for Image 2
    input2 = None
    if file2:
        if not validate_file(file2):
            raise HTTPException(status_code=400, detail="Invalid file type for file2. Only JPEG/PNG allowed.")
        input2 = await file2.read()
    elif image2:
        input2 = image2
    else:
        raise HTTPException(status_code=400, detail="Image 2 is missing. Provide 'file2' or 'image2' (url/path/base64).")
    
    try:
        # result is a dict with similarity, match, and optional error
        result = face_service.compare_faces(input1, input2, strategy=strategy)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
