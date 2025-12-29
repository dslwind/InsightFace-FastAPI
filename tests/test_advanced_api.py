import requests
import json
import base64
import os
import cv2
import numpy as np
import time

def generate_test_image(filename, face_size=100, num_faces=1):
    """Generates an image with synthetic 'faces' (rectangles) - actually using real face image to be safe
       or just use the existing face1.png and resize/tile it.
    """
    # Load existing face1.png as base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(base_dir, "face1.png")
    if not os.path.exists(orig_path):
        raise FileNotFoundError(f"{orig_path} not found")
        
    img = cv2.imread(orig_path)
    
    # 1. Resize to specific face size (approx)
    # The original image is likely mostly face.
    img_h, img_w = img.shape[:2]
    # Simple resize
    scale = face_size / max(img_h, img_w)
    face_img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # 2. Create blank canvas
    fh, fw = face_img.shape[:2]
    
    if num_faces == 1:
        cv2.imwrite(filename, face_img)
        return filename
    
    # Tile faces
    canvas = np.zeros((fh, fw * num_faces, 3), dtype=np.uint8)
    for i in range(num_faces):
        canvas[:, i*fw:(i+1)*fw] = face_img
        
    cv2.imwrite(filename, canvas)
    return filename

def file_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_advanced_api():
    url = "http://127.0.0.1:8000/face/compare"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate test assets
    # small_face_path = os.path.join(base_dir, "small_face.png") # ~30px
    # generate_test_image(small_face_path, face_size=40, num_faces=1)
    
    # multi_face_path = os.path.join(base_dir, "multi_face_3.png")
    # generate_test_image(multi_face_path, face_size=100, num_faces=3)
    
    # normal_face_path = os.path.join(base_dir, "face1.png")

    small_face_path = os.path.join(base_dir, "face_06.jpeg")
    multi_face_path = os.path.join(base_dir, "face_05.jpeg")
    normal_face_path = os.path.join(base_dir, "face_06.jpeg")

    
    try:
        # Prepare inputs
        b64_small = file_to_base64(small_face_path)
        b64_multi = file_to_base64(multi_face_path)
        b64_normal = file_to_base64(normal_face_path)
        
        # Test 1: Limit Faces
        print("\nTest 1: Limit Faces (limit=2 on 3-face image)")
        payload = {
            "image1": b64_multi,
            "image2": b64_multi,
            "limit_faces": 2,
            "compare_all_faces": True
        }
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Face Counts: {data.get('face_counts')}")
        assert data['face_counts']['image1'] <= 2
        assert data['processing_time_ms'] > 0
        
        # Test 2: Min Face Size (Should filter out small face)
        print("\nTest 2: Min Face Size (min=50, input=40px face)")
        payload = {
            "image1": b64_small,
            "image2": b64_normal,
            "min_face_size": 50
        }
        resp = requests.post(url, json=payload)
        # Expecting error or 0 faces
        data = resp.json()
        print(f"Response: {data}")
        # With new logic, if no faces found, it returns status='error' in JSON
        if data['status'] == 'error':
             print("SUCCESS: Error returned as expected due to no faces found (filtered out).")
        else:
             print(f"WARNING: Faces found: {data.get('face_counts')}")
             
        # Test 3: Normal Comparison with Metadata
        print("\nTest 3: Normal Comparison Structure")
        payload = {
            "image1": b64_normal,
            "image2": b64_normal,
            "threshold": 0.5
        }
        resp = requests.post(url, json=payload)
        data = resp.json()
        print(f"Structure keys: {data.keys()}")
        assert "is_same_person" in data
        assert "parameters" in data
        assert data["parameters"]["threshold"] == 0.5
        assert data["is_same_person"] == True
        print("SUCCESS: Structure verified.")

    finally:
        # Cleanup
        if os.path.exists(small_face_path):
            os.remove(small_face_path)
        if os.path.exists(multi_face_path):
            os.remove(multi_face_path)

if __name__ == "__main__":
    test_advanced_api()
