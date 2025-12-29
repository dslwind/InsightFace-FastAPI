import requests
import base64
import os
import json

def test_flexible_inputs():
    url = "http://127.0.0.1:8000/face/compare"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(base_dir, "face1.png")
    img2_path = os.path.join(base_dir, "face2.png")

    if not os.path.exists(img1_path):
        print("Images not found")
        return
        
    with open(img1_path, "rb") as f:
        b64_1 = base64.b64encode(f.read()).decode('utf-8')
    with open(img2_path, "rb") as f:
        b64_2 = base64.b64encode(f.read()).decode('utf-8')

    # Test Base64 (Raw and Data URI)
    print("Testing Base64 input...")
    
    # 1. Raw Base64
    payload_raw = {
        "image1": b64_1,
        "image2": b64_2
    }
    try:
        resp = requests.post(url, json=payload_raw)
        print(f"Raw Base64 Response: {resp.status_code}")
        if resp.status_code == 200:
             print("SUCCESS")
        else:
             print(f"FAILURE: {resp.text}")
             
        # 2. Data URI
        b64_1_prefix = "data:image/png;base64," + b64_1
        payload_uri = {
            "image1": b64_1_prefix,
            "image2": b64_2 # Mixed raw and uri
        }

        resp = requests.post(url, json=payload_uri)
        print(f"Data URI Base64 Response: {resp.status_code}")
        if resp.status_code == 200:
             print("SUCCESS")
        else:
             print(f"FAILURE: {resp.text}")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_flexible_inputs()
