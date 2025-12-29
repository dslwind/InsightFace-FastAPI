import cv2
import numpy as np
import requests
import base64
import os

from insightface.app import FaceAnalysis
from numpy.linalg import norm
from typing import Optional, Dict, Any, Union


class FaceService:
    def __init__(self):
        # using 'buffalo_l' model pack which includes detection and recognition models
        # providers defaults to CPU to avoid complexity with CUDA requirements on user machine
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _load_image(self, image_source: Union[bytes, str]) -> Optional[np.ndarray]:
        img = None
        
        try:
            # 1. Bytes
            if isinstance(image_source, bytes):
                nparr = np.frombuffer(image_source, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
            # 2. String (URL, Path, Base64)
            elif isinstance(image_source, str):
                # URL
                if image_source.startswith(('http://', 'https://')):
                    resp = requests.get(image_source, timeout=10)
                    resp.raise_for_status()
                    nparr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # File Path
                elif os.path.isfile(image_source):
                    img = cv2.imread(image_source)
                    
                # Base64
                else:
                    # check for header like data:image/jpeg;base64,
                    if ',' in image_source:
                        _, encoded = image_source.split(',', 1)
                    else:
                        encoded = image_source
                    
                    decoded = base64.b64decode(encoded)
                    nparr = np.frombuffer(decoded, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
            
        return img


    def get_embedding(self, image_input: Union[bytes, str], strategy: str = "largest") -> Optional[np.ndarray]:
        img = self._load_image(image_input)

        if img is None:
            return None
            
        faces = self.app.get(img)
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
                return (face_cx - cx)**2 + (face_cy - cy)**2

            faces = sorted(faces, key=dist_to_center) # Ascending order (smallest distance)

        elif strategy == "score":
            # Select face with highest detection score (confidence)
            # Assuming 'det_score' exists on face object, or fallback if property name differs
            # InsightFace typical attribute is 'det_score'
            faces = sorted(faces, key=lambda x: getattr(x, 'det_score', 0), reverse=True)
            
        else:
            # Default: largest area
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)

        return faces[0].normed_embedding


    def compare_faces(self, img1_input: Union[bytes, str], img2_input: Union[bytes, str], threshold: float = 0.5, strategy: str = "largest") -> Dict[str, Any]:
        emb1 = self.get_embedding(img1_input, strategy=strategy)

        if emb1 is None:
            return {"error": "No face detected in the first image", "similarity": 0.0, "match": False}
            
        emb2 = self.get_embedding(img2_input, strategy=strategy)


        if emb2 is None:
            return {"error": "No face detected in the second image", "similarity": 0.0, "match": False}
        
        # Compute cosine similarity
        # embeddings are already normed by insightface (normed_embedding), but safe to re-norm or just dot product if guaranteed.
        # InsightFace's normed_embedding is length 1.
        
        sim = np.dot(emb1, emb2) 
        # Clip similarity to [-1, 1] range to handle float precision
        sim = np.clip(sim, -1.0, 1.0)
        
        return {
            "similarity": float(sim),
            "match": bool(sim > threshold)
        }

# Singleton instance
face_service = FaceService()
