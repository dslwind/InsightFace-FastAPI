import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.schemas import FaceCompareRequest, FaceCompareResponse
from app.services.face_service import face_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face", tags=["face"])


@router.post("/compare", response_model=FaceCompareResponse)
async def compare_faces(request: FaceCompareRequest):
    logger.info("Received face comparison request")
    try:
        result = await run_in_threadpool(face_service.compare_faces, **request.model_dump())

        logger.info(
            "Request processed successfully. Time: %sms",
            result.get("processing_time_ms"),
        )

        if result.get("status") == "success":
            return FaceCompareResponse(code=200, msg="success", data=result)

        return FaceCompareResponse(
            code=-1,
            msg=result.get("error_message", "Unknown error"),
            data=result if "processing_time_ms" in result else None,
        )
    except Exception as e:
        logger.error(f"Unexpected error in /face/compare: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
