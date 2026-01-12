from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.logger import setup_logging
from app.routers import face


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

app.include_router(face.router)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Face Comparison API. POST to /face/compare to compare two images."
    }
