from fastapi import FastAPI
from app.routers import face

app = FastAPI(title="Face Comparison API", description="API for comparing faces using InsightFace", version="1.0.0")

app.include_router(face.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Face Comparison API. POST to /face/compare to compare two images."}
