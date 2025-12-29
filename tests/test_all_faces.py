import requests
import os
import cv2
import numpy as np

def test_compare_all_faces():
    url = "http://127.0.0.1:8000/face/compare"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(base_dir, "face1.png")
    img2_path = os.path.join(base_dir, "face2.png")
    
    if not os.path.exists(img1_path):
        print("Images not found")
        return

    # Create a composite image with 2 faces (side by side)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Resize to same height if needed
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_h = max(h1, h2)
    
    if h1 != max_h:
        img1 = cv2.resize(img1, (int(w1 * max_h / h1), max_h))
    if h2 != max_h:
        img2 = cv2.resize(img2, (int(w2 * max_h / h2), max_h))
        
    # Concatenate
    multi_face_img = np.hstack((img1, img2))
    multi_face_path = os.path.join(base_dir, "multi_face_temp.png")
    cv2.imwrite(multi_face_path, multi_face_img)
    
    try:
        print(f"Created composite image at {multi_face_path}")
        
        # Test 1: Multi-face vs Face 1 (Should match)
        print("\nTest 1: Multi-face vs Face 1 (compare_all_faces=True)")
        data = {
            "image1": multi_face_path,
            "image2": img1_path,
            "compare_all_faces": True
        }
        resp = requests.post(url, data=data)
        print(f"Response: {resp.status_code}, {resp.json()}")
        assert resp.json()['match'] == True, "Should match face1 in multi-face image"

        # Test 2: Multi-face vs Face 2 (Should match)
        print("\nTest 2: Multi-face vs Face 2 (compare_all_faces=True)")
        data = {
            "image1": multi_face_path,
            "image2": img2_path,
            "compare_all_faces": True
        }
        resp = requests.post(url, data=data)
        print(f"Response: {resp.status_code}, {resp.json()}")
        assert resp.json()['match'] == True, "Should match face2 in multi-face image"
        
        # Test 3: Face 1 vs Face 2 with compare_all_faces=True (Should match? No, different people)
        # Assuming face1 and face2 are different people.
        print("\nTest 3: Face 1 vs Face 2 (compare_all_faces=True)")
        data = {
            "image1": img1_path,
            "image2": img2_path,
            "compare_all_faces": True
        }
        resp = requests.post(url, data=data)
        print(f"Response: {resp.status_code}, {resp.json()}")
        # We don't assert boolean here as we don't know if they are same person, but structure check is good.
        
        print("\nAll tests passed!")

    finally:
        # Cleanup
        if os.path.exists(multi_face_path):
            os.remove(multi_face_path)
            print(f"\nRemoved temp file {multi_face_path}")

if __name__ == "__main__":
    test_compare_all_faces()
