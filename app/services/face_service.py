import base64
import logging
import os
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis
from numpy.linalg import norm

from app.config import settings

logger = logging.getLogger(__name__)


class FaceService:

    def __init__(self):
        # using model pack from settings
        # providers defaults to CPU or CUDA based on settings
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if settings.ENABLE_CUDA
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL_NAME, providers=providers
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _load_image(
        self, image_source: Union[bytes, str, np.ndarray]
    ) -> Optional[np.ndarray]:
        if isinstance(image_source, np.ndarray):
            return image_source

        img = None

        try:
            # 1. Bytes
            if isinstance(image_source, bytes):
                nparr = np.frombuffer(image_source, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 2. String (URL, Base64)
            elif isinstance(image_source, str):
                # URL
                if image_source.startswith(("http://", "https://")):
                    resp = requests.get(image_source, timeout=10)
                    resp.raise_for_status()
                    nparr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Base64
                else:
                    # check for header like data:image/jpeg;base64,
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

                # Removed direct file path support for security/deployment context
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
    ) -> Optional[Any]:
        # Returns list of face objects
        img = self._load_image(image_input)
        if img is None:
            return None

        faces = self.app.get(img)
        if not faces:
            return None

        # Filter by min_face_size
        if min_face_size > 0:
            faces = [
                f
                for f in faces
                if (f.bbox[2] - f.bbox[0]) >= min_face_size
                and (f.bbox[3] - f.bbox[1]) >= min_face_size
            ]

        # Filter by detection_threshold
        if detection_threshold > 0:
            # assuming det_score exists
            faces = [
                f for f in faces if getattr(f, "det_score", 0) >= detection_threshold
            ]

        if not faces:
            return None

        # Sort according to strategy
        if strategy == "center":
            h, w = img.shape[:2]
            cx, cy = w / 2, h / 2

            def dist_to_center(face):
                box = face.bbox
                face_cx = (box[0] + box[2]) / 2
                face_cy = (box[1] + box[3]) / 2
                return (face_cx - cx) ** 2 + (face_cy - cy) ** 2

            faces = sorted(faces, key=dist_to_center)
        elif strategy == "confidence" or strategy == "score":
            faces = sorted(
                faces, key=lambda x: getattr(x, "det_score", 0), reverse=True
            )
        else:
            # defaults to area/largest
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )

        # Apply limit
        if limit_faces > 0:
            faces = faces[:limit_faces]

        return faces

    def get_embedding(
        self, image_input: Union[bytes, str], strategy: str = "largest"
    ) -> Optional[np.ndarray]:
        # Legacy method support, uses default filters
        faces = self.get_faces(image_input, strategy=strategy)
        if not faces:
            return None

        # Strategies
        if strategy == "center":
            # Select face closest to image center
            h, w = img.shape[:2]
            cx, cy = w / 2, h / 2

            def dist_to_center(face):
                # face.bbox is [x1, y1, x2, y2]
                box = face.bbox
                face_cx = (box[0] + box[2]) / 2
                face_cy = (box[1] + box[3]) / 2
                return (face_cx - cx) ** 2 + (face_cy - cy) ** 2

            faces = sorted(
                faces, key=dist_to_center
            )  # Ascending order (smallest distance)

        elif strategy == "score":
            # Select face with highest detection score (confidence)
            # Assuming 'det_score' exists on face object, or fallback if property name differs
            # InsightFace typical attribute is 'det_score'
            faces = sorted(
                faces, key=lambda x: getattr(x, "det_score", 0), reverse=True
            )

        else:
            # Default: largest area
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )

        return faces[0].normed_embedding

    def compare_faces(
        self,
        image1: str,
        image2: str,
        threshold: float = settings.DEFAULT_THRESHOLD,
        limit_faces: int = settings.DEFAULT_LIMIT_FACES,
        min_face_size: int = settings.DEFAULT_MIN_FACE_SIZE,
        return_face_data: bool = settings.DEFAULT_RETURN_FACE_DATA,
        return_landmarks: bool = settings.DEFAULT_RETURN_LANDMARKS,
        detection_threshold: float = settings.DEFAULT_DETECTION_THRESHOLD,
        best_face_strategy: str = settings.DEFAULT_BEST_FACE_STRATEGY,
        input_format: str = settings.DEFAULT_INPUT_FORMAT,
        compare_all_faces: bool = settings.DEFAULT_COMPARE_ALL_FACES,
        enable_rotation: bool = settings.DEFAULT_ENABLE_ROTATION,
    ) -> Dict[str, Any]:

        import time

        start_time = time.time()

        # Prepare response structure
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
                "input_format": input_format,
                "enable_rotation": enable_rotation,
            },
        }

        try:
            # 1. Load Image 2 and detect faces (once)
            faces2 = self.get_faces(
                image2,
                min_face_size=min_face_size,
                detection_threshold=detection_threshold,
                limit_faces=limit_faces,
                strategy=best_face_strategy,
            )
            response["face_counts"]["image2"] = len(faces2) if faces2 else 0
            logger.debug(f"Image 2: Found {response['face_counts']['image2']} faces")

            if not faces2:
                response["status"] = "error"
                response["error_message"] = "No face detected in image2"
                logger.warning("No face detected in image2")
            else:
                # 2. Load Image 1
                img1 = self._load_image(image1)
                if img1 is None:
                    response["status"] = "error"
                    response["error_message"] = "Could not load image1"
                else:
                    # 3. Rotation Loop for Image 1
                    # max 4 trials: 0, 90, 180, 270 degrees
                    max_trials = 4 if enable_rotation else 1
                    current_img1 = img1

                    for trial in range(max_trials):
                        if trial > 0:
                            logger.info(
                                f"Retrying with 90 degree counter-clockwise rotation (Trial {trial})"
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

                        num_faces1 = len(faces1) if faces1 else 0
                        response["face_counts"]["image1"] = num_faces1

                        if not faces1:
                            logger.debug(f"Trial {trial}: No face detected in image1")
                            continue

                        # Prepare lists for comparison
                        list1 = faces1 if compare_all_faces else [faces1[0]]
                        list2 = faces2 if compare_all_faces else [faces2[0]]

                        max_sim = -1.0
                        for f1 in list1:
                            emb1 = f1.normed_embedding
                            for f2 in list2:
                                emb2 = f2.normed_embedding
                                sim = np.dot(emb1, emb2)
                                sim = np.clip(sim, -1.0, 1.0)
                                if sim > max_sim:
                                    max_sim = sim

                        response["similarity_score"] = float(max_sim)
                        response["is_same_person"] = bool(max_sim > threshold)

                        if response["is_same_person"]:
                            logger.info(
                                f"Match found at rotation trial {trial}. Similarity: {max_sim:.4f}"
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
            logger.error(f"Error comparing faces: {e}")

        # Calculate time
        end_time = time.time()
        response["processing_time_ms"] = float((end_time - start_time) * 1000)

        return response


# Singleton instance
face_service = FaceService()
