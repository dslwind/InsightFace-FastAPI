import asyncio
import logging
import sys
import types
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


fake_insightface = types.ModuleType("insightface")
fake_insightface_app = types.ModuleType("insightface.app")


class DummyFaceAnalysis:
    def __init__(self, *args, **kwargs):
        pass

    def prepare(self, *args, **kwargs):
        pass

    def get(self, img):
        return []


fake_insightface_app.FaceAnalysis = DummyFaceAnalysis
fake_insightface.app = fake_insightface_app
sys.modules.setdefault("insightface", fake_insightface)
sys.modules.setdefault("insightface.app", fake_insightface_app)

from app.routers.face import compare_faces, detect_faces
from app.schemas import FaceCompareRequest, FaceDetectRequest
from app.services.face_service import FaceService


class DummyFace:
    def __init__(self, bbox, det_score, normed_embedding=None):
        self.bbox = bbox
        self.det_score = det_score
        self.normed_embedding = (
            np.array(normed_embedding, dtype=np.float32)
            if normed_embedding is not None
            else np.array([1.0, 0.0], dtype=np.float32)
        )


def test_detect_faces_reports_total_count_before_limit(monkeypatch):
    service = FaceService()
    all_faces = [
        DummyFace([0, 0, 40, 40], 0.99),
        DummyFace([50, 0, 90, 40], 0.95),
        DummyFace([100, 0, 140, 40], 0.90),
    ]

    monkeypatch.setattr(service, "initialize", lambda: None)
    monkeypatch.setattr(
        service,
        "_load_image",
        lambda _: np.zeros((80, 160, 3), dtype=np.uint8),
    )

    def fake_get_faces(image_input, **kwargs):
        assert kwargs["limit_faces"] == 0
        return all_faces

    monkeypatch.setattr(service, "get_faces", fake_get_faces)

    result = service.detect_faces("image-data", limit_faces=2)

    assert result["status"] == "success"
    assert result["face_count"] == 3
    assert len(result["faces"]) == 2
    assert [face["index"] for face in result["faces"]] == [1, 2]


def test_detect_route_logs_failure_as_warning(monkeypatch, caplog):
    result = {
        "status": "error",
        "error_message": "Could not load image",
        "processing_time_ms": 12.3,
        "face_count": 0,
        "image_size": {"width": 0, "height": 0},
        "faces": [],
        "parameters": {
            "detection_threshold": 0.6,
            "limit_faces": 0,
            "min_face_size": 0,
            "best_face_strategy": "center",
        },
    }

    async def fake_run_in_threadpool(func, **kwargs):
        return result

    monkeypatch.setattr(
        "app.routers.face.run_in_threadpool",
        fake_run_in_threadpool,
    )

    with caplog.at_level(logging.INFO, logger="app.routers.face"):
        response = asyncio.run(detect_faces(FaceDetectRequest(image="bad-image")))

    assert response.code == -1
    assert "processed successfully" not in caplog.text
    assert any(
        record.levelno == logging.WARNING and "Face detection request failed" in record.getMessage()
        for record in caplog.records
    )


def test_compare_faces_echoes_compare_all_faces(monkeypatch):
    service = FaceService()
    dummy_image = np.zeros((80, 160, 3), dtype=np.uint8)
    faces = [
        DummyFace([0, 0, 40, 40], 0.99, normed_embedding=[1.0, 0.0]),
        DummyFace([50, 0, 90, 40], 0.95, normed_embedding=[0.8, 0.2]),
    ]

    monkeypatch.setattr(service, "initialize", lambda: None)
    monkeypatch.setattr(service, "_load_image", lambda _: dummy_image)
    monkeypatch.setattr(service, "get_faces", lambda *args, **kwargs: faces)

    result = service.compare_faces("image-1", "image-2", compare_all_faces=True)

    assert result["status"] == "success"
    assert result["parameters"]["compare_all_faces"] is True
    assert result["face_counts"] == {"image1": 2, "image2": 2}


def test_compare_route_logs_failure_as_warning(monkeypatch, caplog):
    result = {
        "status": "error",
        "error_message": "No face detected in image2",
        "processing_time_ms": 22.4,
        "is_same_person": False,
        "similarity_score": -1.0,
        "face_counts": {"image1": 0, "image2": 0},
        "parameters": {
            "threshold": 0.3,
            "detection_threshold": 0.6,
            "limit_faces": 0,
            "min_face_size": 0,
            "best_face_strategy": "center",
            "compare_all_faces": False,
            "enable_rotation": False,
        },
    }

    async def fake_run_in_threadpool(func, **kwargs):
        return result

    monkeypatch.setattr(
        "app.routers.face.run_in_threadpool",
        fake_run_in_threadpool,
    )

    with caplog.at_level(logging.INFO, logger="app.routers.face"):
        response = asyncio.run(
            compare_faces(FaceCompareRequest(image1="a", image2="b"))
        )

    assert response.code == -1
    assert "processed successfully" not in caplog.text
    assert any(
        record.levelno == logging.WARNING and "Face comparison request failed" in record.getMessage()
        for record in caplog.records
    )
