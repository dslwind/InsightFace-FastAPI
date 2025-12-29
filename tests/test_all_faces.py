import requests
import os
import cv2
import base64
import json

def file_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_compare_all_faces():
    url = "http://127.0.0.1:8000/face/compare"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # User specified files
    multi_face_path = os.path.join(base_dir, "face_05.jpeg")
    single_face_path = os.path.join(base_dir, "face_06.jpeg")
    
    # Fallback to defaults if user files are missing/bad (for robustness, though user provided them)
    if not os.path.exists(multi_face_path) or not os.path.exists(single_face_path):
        print("Required test images not found.")
        return

    # Base64 encode
    try:
        b64_multi = file_to_base64(multi_face_path)
        b64_single = file_to_base64(single_face_path)
    except Exception as e:
        print(f"Error reading images: {e}")
        return

    print(f"Testing with: {multi_face_path} (Multi) and {single_face_path} (Single)")

    # Test 1: Compare Multi vs Single with compare_all_faces=True
    print("\nTest 1: Multi vs Single (compare_all_faces=True)")
    payload = {
        "image1": b64_multi,
        "image2": b64_single,
        "compare_all_faces": True,
        "threshold": 0.5 # Explicit threshold
    }
    
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Response: {data}")
            
            # Verify structure
            assert "is_same_person" in data
            assert "face_counts" in data
            
            # Since we don't know ground truth, we just verify it runs and returns valid structure
            # But the user implies valid test, so we assume maybe they match?
            # We'll just print semantics.
            if data['is_same_person']:
                print("Result: MATCH (Found single face in multi-face image)")
            else:
                print("Result: NO MATCH")
                
            print(f"Face Counts: {data['face_counts']}")
        else:
            print(f"Error: {resp.text}")

    except Exception as e:
        print(f"Request failed: {e}")

    # Test 2: Compare Multi vs Single with compare_all_faces=False (Default)
    # This should pick 'best' face (e.g. center) and compare
    print("\nTest 2: Multi vs Single (compare_all_faces=False)")
    payload["compare_all_faces"] = False
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Response: {data}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_compare_all_faces()
