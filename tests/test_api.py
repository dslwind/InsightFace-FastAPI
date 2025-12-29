import requests
import os

def test_compare():
    url = "http://127.0.0.1:8000/face/compare"
    
    # Ensure images exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(base_dir, "face1.png")
    img2_path = os.path.join(base_dir, "face2.png")
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Test images not found. Please ensure face1.png and face2.png are in the directory.")
        return

    files = [
        ('file1', ('face1.png', open(img1_path, 'rb'), 'image/png')),
        ('file2', ('face2.png', open(img2_path, 'rb'), 'image/png'))
    ]
    
    try:
        response = requests.post(url, files=files)
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
