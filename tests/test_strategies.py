import requests
import os
import base64
import json

def file_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_strategies():
    url = "http://127.0.0.1:8000/face/compare"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(base_dir, "face1.png")
    img2_path = os.path.join(base_dir, "face2.png")

    if not os.path.exists(img1_path):
        print("Images not found")
        return
        
    b64_1 = file_to_base64(img1_path)
    b64_2 = file_to_base64(img2_path)

    print("Testing Strategy: largest (default)...")
    payload_largest = {
        "image1": b64_1,
        "image2": b64_2,
        "best_face_strategy": "area" # mapped from legacy "largest" or new "area"
    }
    resp = requests.post(url, json=payload_largest)
    print(f"Largest Response: {resp.status_code}, {resp.json()}")

    print("Testing Strategy: center...")
    payload_center = {
        "image1": b64_1,
        "image2": b64_2,
        "best_face_strategy": "center"
    }
    resp = requests.post(url, json=payload_center)
    print(f"Center Response: {resp.status_code}, {resp.json()}")
    
    print("Testing Strategy: confidence...")
    payload_score = {
        "image1": b64_1,
        "image2": b64_2,
        "best_face_strategy": "confidence"
    }
    resp = requests.post(url, json=payload_score)
    print(f"Confidence Response: {resp.status_code}, {resp.json()}")

if __name__ == "__main__":
    test_strategies()
