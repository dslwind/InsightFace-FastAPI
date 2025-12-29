import requests
import os

def test_strategies():
    url = "http://127.0.0.1:8000/face/compare"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(base_dir, "face1.png")
    img2_path = os.path.join(base_dir, "face2.png")

    if not os.path.exists(img1_path):
        print("Images not found")
        return

    print("Testing Strategy: largest (default)...")
    data_largest = {
        "image1": img1_path,
        "image2": img2_path,
        "strategy": "largest"
    }
    resp = requests.post(url, data=data_largest)
    print(f"Largest Response: {resp.status_code}, {resp.json()}")

    print("Testing Strategy: center...")
    data_center = {
        "image1": img1_path,
        "image2": img2_path,
        "strategy": "center"
    }
    resp = requests.post(url, data=data_center)
    print(f"Center Response: {resp.status_code}, {resp.json()}")
    
    print("Testing Strategy: score...")
    data_score = {
        "image1": img1_path,
        "image2": img2_path,
        "strategy": "score"
    }
    resp = requests.post(url, data=data_score)
    print(f"Score Response: {resp.status_code}, {resp.json()}")

if __name__ == "__main__":
    test_strategies()
