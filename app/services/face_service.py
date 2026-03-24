import base64
import logging
import os
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis

from app.config import settings

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(self):
        self.app: Optional[FaceAnalysis] = None
        self._init_lock = Lock()
        self._root_path = os.path.abspath(settings.INSIGHTFACE_ROOT)

    def _build_providers(self) -> List[Union[str, tuple[str, dict[str, Any]]]]:
        providers: List[Union[str, tuple[str, dict[str, Any]]]] = []

        if settings.ENABLE_CUDA:
            if settings.ENABLE_TENSORRT:
                trt_cache_path = os.path.join(self._root_path, "trt_engines")
                os.makedirs(trt_cache_path, exist_ok=True)
                providers.append(
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": trt_cache_path,
                            "trt_fp16_enable": True,
                        },
                    )
                )

            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")
        return providers

    def initialize(self) -> None:
        if self.app is not None:
            return

        with self._init_lock:
            if self.app is not None:
                return

            logger.info(
                "Initializing InsightFace model pack '%s' (cuda=%s, tensorrt=%s)",
                settings.INSIGHTFACE_MODEL_NAME,
                settings.ENABLE_CUDA,
                settings.ENABLE_TENSORRT,
            )

            app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL_NAME,
                root=self._root_path,
                providers=self._build_providers(),
            )
            app.prepare(
                ctx_id=0 if settings.ENABLE_CUDA else -1,
                det_size=(640, 640),
            )
            self.app = app
            logger.info("InsightFace initialization complete")

    def warmup(self) -> None:
        """
        Perform a dummy inference to warm up the model.
        This ensures the first user request doesn't incur initialization latency.
        """
        self.initialize()

        try:
            logger.info("Starting model warmup...")

            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.app.get(dummy_img)

            if "recognition" in self.app.models:
                rec_model = self.app.models["recognition"]
                dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)

                if hasattr(rec_model, "session"):
                    input_name = rec_model.session.get_inputs()[0].name
                    rec_model.session.run(None, {input_name: dummy_input})
                    logger.info("Recognition model warmup completed (Direct ONNX).")
                else:
                    logger.warning(
                        "Recognition model has no 'session' attribute, skipping specific warmup."
                    )

            logger.info("Model warmup completed.")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _load_image(
        self, image_source: Union[bytes, str, np.ndarray]
    ) -> Optional[np.ndarray]:
        if isinstance(image_source, np.ndarray):
            return image_source

        img = None

        try:
            if isinstance(image_source, bytes):
                nparr = np.frombuffer(image_source, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(image_source, str):
                if image_source.startswith(("http://", "https://")):
                    resp = requests.get(image_source, timeout=10)
                    resp.raise_for_status()
                    nparr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    if "," in image_source:
                        _, encoded = image_source.split(",", 1)
                    else:
                        encoded = image_source

                    try:
                        decoded = base64.b64decode(encoded)
                        nparr = np.frombuffer(decoded, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    except Exception as e:
                        logger.error(f"Base64 decode error: {e}")
                        return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

        return img

    def get_faces(
        self,
        image_input: Union[bytes, str, np.ndarray],
        min_face_size: int = 0,
        detection_threshold: float = 0.0,
        limit_faces: int = 0,
        strategy: str = "largest",
    ) -> Optional[List[Any]]:
        self.initialize()

        img = self._load_image(image_input)
        if img is None:
            return None

        faces = self.app.get(img)
        if not faces:
            return None

        if min_face_size > 0:
            faces = [
                f
                for f in faces
                if (f.bbox[2] - f.bbox[0]) >= min_face_size
                and (f.bbox[3] - f.bbox[1]) >= min_face_size
            ]

        if detection_threshold > 0:
            faces = [
                f for f in faces if getattr(f, "det_score", 0) >= detection_threshold
            ]

        if not faces:
            return None

        normalized_strategy = strategy.lower()
        if normalized_strategy == "center":
            h, w = img.shape[:2]
            cx, cy = w / 2, h / 2

            def dist_to_center(face: Any) -> float:
                box = face.bbox
                face_cx = (box[0] + box[2]) / 2
                face_cy = (box[1] + box[3]) / 2
                return (face_cx - cx) ** 2 + (face_cy - cy) ** 2

            faces = sorted(faces, key=dist_to_center)
        elif normalized_strategy in {"confidence", "score"}:
            faces = sorted(
                faces, key=lambda x: getattr(x, "det_score", 0), reverse=True
            )
        else:
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )

        if limit_faces > 0:
            faces = faces[:limit_faces]

        return faces

    def get_embedding(
        self, image_input: Union[bytes, str, np.ndarray], strategy: str = "largest"
    ) -> Optional[np.ndarray]:
        faces = self.get_faces(image_input, strategy=strategy)
        if not faces:
            return None

        return faces[0].normed_embedding

    def compare_faces(
        self,
        image1: str,
        image2: str,
        threshold: float = settings.DEFAULT_THRESHOLD,
        limit_faces: int = settings.DEFAULT_LIMIT_FACES,
        min_face_size: int = settings.DEFAULT_MIN_FACE_SIZE,
        detection_threshold: float = settings.DEFAULT_DETECTION_THRESHOLD,
        best_face_strategy: str = settings.DEFAULT_BEST_FACE_STRATEGY,
        compare_all_faces: bool = settings.DEFAULT_COMPARE_ALL_FACES,
        enable_rotation: bool = settings.DEFAULT_ENABLE_ROTATION,
    ) -> Dict[str, Any]:
        import concurrent.futures
        import time

        self.initialize()
        start_time = time.time()

        response = {
            "is_same_person": False,
            "similarity_score": -1.0,
            "status": "success",
            "error_message": None,
            "face_counts": {"image1": 0, "image2": 0},
            "processing_time_ms": 0.0,
            "parameters": {
                "threshold": threshold,
                "detection_threshold": detection_threshold,
                "limit_faces": limit_faces,
                "min_face_size": min_face_size,
                "best_face_strategy": best_face_strategy,
                "enable_rotation": enable_rotation,
            },
        }

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_img1 = executor.submit(self._load_image, image1)
                future_img2 = executor.submit(self._load_image, image2)

                img1 = future_img1.result()
                img2 = future_img2.result()

            if img2 is None:
                response["status"] = "error"
                response["error_message"] = "Could not load image2"
                logger.warning("Could not load image2")
            elif img1 is None:
                response["status"] = "error"
                response["error_message"] = "Could not load image1"
                logger.warning("Could not load image1")
            else:
                faces2 = self.get_faces(
                    img2,
                    min_face_size=min_face_size,
                    detection_threshold=detection_threshold,
                    limit_faces=limit_faces,
                    strategy=best_face_strategy,
                )
                response["face_counts"]["image2"] = len(faces2) if faces2 else 0
                logger.debug(
                    "Image 2: Found %s faces", response["face_counts"]["image2"]
                )

                if not faces2:
                    response["status"] = "error"
                    response["error_message"] = "No face detected in image2"
                    logger.warning("No face detected in image2")
                else:
                    max_trials = 4 if enable_rotation else 1
                    current_img1 = img1

                    for trial in range(max_trials):
                        if trial > 0:
                            logger.info(
                                "Retrying with 90 degree counter-clockwise rotation (Trial %s)",
                                trial,
                            )
                            current_img1 = cv2.rotate(
                                current_img1, cv2.ROTATE_90_COUNTERCLOCKWISE
                            )

                        faces1 = self.get_faces(
                            current_img1,
                            min_face_size=min_face_size,
                            detection_threshold=detection_threshold,
                            limit_faces=limit_faces,
                            strategy=best_face_strategy,
                        )

                        response["face_counts"]["image1"] = len(faces1) if faces1 else 0

                        if not faces1:
                            logger.debug("Trial %s: No face detected in image1", trial)
                            continue

                        list1 = faces1 if compare_all_faces else [faces1[0]]
                        list2 = faces2 if compare_all_faces else [faces2[0]]

                        max_sim = -1.0
                        for f1 in list1:
                            emb1 = f1.normed_embedding
                            for f2 in list2:
                                emb2 = f2.normed_embedding
                                sim = float(np.dot(emb1, emb2))
                                sim = float(np.clip(sim, -1.0, 1.0))
                                if sim > max_sim:
                                    max_sim = sim

                        response["similarity_score"] = max_sim
                        response["is_same_person"] = bool(max_sim > threshold)

                        if response["is_same_person"]:
                            logger.info(
                                "Match found at rotation trial %s. Similarity: %.4f",
                                trial,
                                max_sim,
                            )
                            break

                    if (
                        not response["is_same_person"]
                        and response["face_counts"]["image1"] == 0
                    ):
                        response["status"] = "error"
                        response["error_message"] = (
                            "No face detected in image1 (tried rotations)"
                            if enable_rotation
                            else "No face detected in image1"
                        )

        except Exception as e:
            response["status"] = "error"
            response["error_message"] = str(e)
            logger.error("Error comparing faces: %s", e, exc_info=True)

        response["processing_time_ms"] = float((time.time() - start_time) * 1000)
        return response


face_service = FaceService()
