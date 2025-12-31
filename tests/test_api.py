import requests
import os
import base64

def test_compare():
    url = "http://127.0.0.1:8000/face/compare"
    
    # Ensure images exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(base_dir, "image003.jpeg")
    img2_path = os.path.join(base_dir, "image002.jpeg")
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Test images not found. Please ensure image001.jpeg and image002.jpeg are in the directory.")
        return

    with open(img1_path, "rb") as f:
        b64_1 = base64.b64encode(f.read()).decode('utf-8')
    with open(img2_path, "rb") as f:
        b64_2 = base64.b64encode(f.read()).decode('utf-8')
    
    try:
        response = requests.post(url, json={"image1": b64_1, "image2": b64_2})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            if "similarity" in data and "match" in data:
                print("SUCCESS: API returned valid response structure.")
            else:
                print("FAILURE: Unexpected response structure.")
        else:
            print("FAILURE: API returned non-200 status code.")
            
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")

if __name__ == "__main__":
    test_compare()
