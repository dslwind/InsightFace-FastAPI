import time
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, StreamingResponse

from app.logger import setup_logging
from app.routers import face
from app.database_logger import db_logger
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Setup logging
    setup_logging()
    yield
    # Shutdown events if any

app = FastAPI(
    title="Face Comparison API",
    description="API for comparing faces using InsightFace",
    version="1.0.0",
    lifespan=lifespan,
)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.ENABLE_DB_LOGGING:
            return await call_next(request)

        start_time = time.time()
        
        # 1. Capture Request Body
        request_body_bytes = await request.body()
        
        # Replace the receive channel so the route can still read the body
        async def receive():
            return {"type": "http.request", "body": request_body_bytes}
        request._receive = receive

        # Parse request body for logging
        log_request_body = {}
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                log_request_body = json.loads(request_body_bytes.decode("utf-8"))
                # Truncate large strings (like base64 images) to keep DB clean
                log_request_body = self._truncate_large_values(log_request_body)
            except Exception:
                log_request_body = "<failed to parse json>"
        else:
            log_request_body = f"<Non-JSON content: {content_type}>"

        # 2. Call Next
        response = await call_next(request)
        
        # 3. Capture Response Body
        response_body = b""
        # We need to consume the iterator to log it
        async for chunk in response.body_iterator:
            response_body += chunk
            
        process_time = time.time() - start_time
        
        # Parse response body for logging
        log_response_body = {}
        res_content_type = response.headers.get("content-type", "")
        if "application/json" in res_content_type:
            try:
                log_response_body = json.loads(response_body.decode("utf-8"))
            except Exception:
                log_response_body = "<failed to parse response json>"
        else:
            log_response_body = f"<Non-JSON response: {res_content_type}>"

        # 4. Log to DB
        db_logger.log_request(
            client_ip=request.client.host if request.client else "unknown",
            method=request.method,
            path=request.url.path,
            params={
                "query": dict(request.query_params),
                "body": log_request_body
            },
            status=response.status_code,
            response=log_response_body,
            duration=process_time
        )
        
        # 5. Return a new response with the captured body
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )

    def _truncate_large_values(self, data):
        """Recursively truncate long strings in a dictionary/list."""
        if isinstance(data, dict):
            return {k: self._truncate_large_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._truncate_large_values(i) for i in data]
        elif isinstance(data, str) and len(data) > 500:
            return data[:100] + "... [truncated, total length: " + str(len(data)) + "]"
        return data

app.add_middleware(LoggingMiddleware)
app.include_router(face.router)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Face Comparison API. POST to /face/compare to compare two images."
    }
