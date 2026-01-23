import logging

from fastapi import APIRouter, Body, HTTPException

from app.schemas import FaceCompareRequest, FaceCompareResponse
from app.services.face_service import face_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face", tags=["face"])


@router.post("/compare", response_model=FaceCompareResponse)
async def compare_faces(request: FaceCompareRequest):
    logger.info("Received face comparison request")
    try:
        from fastapi.concurrency import run_in_threadpool

        # Pass request fields as keyword arguments to the service
        # Run in threadpool to avoid blocking the event loop
        result = await run_in_threadpool(face_service.compare_faces, **request.model_dump())
        
        logger.info(
            f"Request processed successfully. Time: {result.get('processing_time_ms')}ms"
        )

        if result.get("status") == "success":
            return FaceCompareResponse(
                code=200,
                msg="success",
                data=result
            )
        else:
            return FaceCompareResponse(
                code=-1,
                msg=result.get("error_message", "Unknown error"),
                data=result if "processing_time_ms" in result else None
            )
    except Exception as e:
        # In case of unexpected server error not handled by service
        logger.error(f"Unexpected error in /face/compare: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
