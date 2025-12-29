import requests
import base64
import os

def test_flexible_inputs():
    url = "http://127.0.0.1:8000/face/compare"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(base_dir, "face1.png")
    img2_path = os.path.join(base_dir, "face2.png")

    if not os.path.exists(img1_path):
        print("Images not found")
        return

    # 1. Test Path
    print("Testing Path input...")
    data_path = {
        "image1": img1_path,
        "image2": img2_path
    }
    resp = requests.post(url, data=data_path)
    print(f"Path Response: {resp.status_code}, {resp.json()}")

    # 2. Test Base64
    print("Testing Base64 input...")
    with open(img1_path, "rb") as f:
        b64_1 = base64.b64encode(f.read()).decode('utf-8')
    with open(img2_path, "rb") as f:
        b64_2 = base64.b64encode(f.read()).decode('utf-8')
    
    # Add prefix for robustness test
    b64_1_prefix = "data:image/png;base64," + b64_1

    data_b64 = {
        "image1": b64_1_prefix,
        "image2": b64_2
    }
    resp = requests.post(url, data=data_b64)
    print(f"Base64 Response: {resp.status_code}, {resp.json()}")
    
    # 3. Test Mixed (File + Base64)
    print("Testing Mixed input...")
    # Note: When using 'files', 'data' fields are sent as multipart form fields too.
    files = {
        'file1': ('face1.png', open(img1_path, 'rb'), 'image/png')
    }
    data_mixed = {
        "image2": b64_2
    }
    resp = requests.post(url, files=files, data=data_mixed)
    print(f"Mixed Response: {resp.status_code}, {resp.json()}")

if __name__ == "__main__":
    test_flexible_inputs()
