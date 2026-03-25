import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config import settings
from app.database_logger import db_logger
from app.logger import setup_logging
from app.routers import face


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    from app.services.face_service import face_service

    face_service.initialize()
    face_service.warmup()

    yield


app = FastAPI(
    title="InsightFace Detection and Comparison API",
    description="API for detecting and comparing faces using InsightFace",
    version="1.0.0",
    lifespan=lifespan,
)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.ENABLE_DB_LOGGING:
            return await call_next(request)

        start_time = time.time()

        request_body_bytes = await request.body()

        async def receive():
            return {"type": "http.request", "body": request_body_bytes}

        request._receive = receive

        log_request_body = {}
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                log_request_body = json.loads(request_body_bytes.decode("utf-8"))
                log_request_body = self._truncate_large_values(log_request_body)
            except Exception:
                log_request_body = "<failed to parse json>"
        else:
            log_request_body = f"<Non-JSON content: {content_type}>"

        response = await call_next(request)

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        process_time = time.time() - start_time

        log_response_body = {}
        res_content_type = response.headers.get("content-type", "")
        if "application/json" in res_content_type:
            try:
                log_response_body = json.loads(response_body.decode("utf-8"))
            except Exception:
                log_response_body = "<failed to parse response json>"
        else:
            log_response_body = f"<Non-JSON response: {res_content_type}>"

        db_logger.log_request(
            client_ip=request.client.host if request.client else "unknown",
            method=request.method,
            path=request.url.path,
            params={"query": dict(request.query_params), "body": log_request_body},
            status=response.status_code,
            response=log_response_body,
            duration=process_time,
        )

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    def _truncate_large_values(self, data):
        if isinstance(data, dict):
            return {k: self._truncate_large_values(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._truncate_large_values(i) for i in data]
        if isinstance(data, str) and len(data) > 500:
            return data[:100] + "... [truncated, total length: " + str(len(data)) + "]"
        return data


app.add_middleware(LoggingMiddleware)
app.include_router(face.router)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the InsightFace API. Use /face/detect to detect faces and /face/compare to compare two images."
    }
