import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from typing import Optional, Dict, Any

class FaceService:
    def __init__(self):
        # using 'buffalo_l' model pack which includes detection and recognition models
        # providers defaults to CPU to avoid complexity with CUDA requirements on user machine
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _process_image(self, img_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def get_embedding(self, img_bytes: bytes) -> Optional[np.ndarray]:
        img = self._process_image(img_bytes)
        if img is None:
            return None
            
        faces = self.app.get(img)
        if not faces:
            return None
        
        # Sort faces by area (width * height) descending and pick the largest
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        return faces[0].normed_embedding

    def compare_faces(self, img1_bytes: bytes, img2_bytes: bytes, threshold: float = 0.5) -> Dict[str, Any]:
        emb1 = self.get_embedding(img1_bytes)
        if emb1 is None:
            return {"error": "No face detected in the first image", "similarity": 0.0, "match": False}
            
        emb2 = self.get_embedding(img2_bytes)
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
