import requests
import json
import base64
import os
import time

# This test assumes the server is running on localhost:8000
# Usage: Start server first, then run this.

def test_logging():
    url = "http://127.0.0.1:8000/face/compare"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "face1.png")
    
    if not os.path.exists(img_path):
        print("Skipping request: face1.png not found")
    else: 
        with open(img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
            
        payload = {
            "image1": b64_img,
            "image2": b64_img
        }
        
        try:
             requests.post(url, json=payload)
             print("Request sent.")
        except Exception as e:
             print(f"Request failed (server might be down): {e}")

    # Verify Log File
    log_file = os.path.abspath(os.path.join(base_dir, "../logs/insightface_api.log"))
    if os.path.exists(log_file):
        print(f"Log file found at: {log_file}")
        with open(log_file, "r") as f:
            lines = f.readlines()
            print(f"Log lines count: {len(lines)}")
            if len(lines) > 0:
                print("Last log line:", lines[-1].strip())
                if "Comparison complete" in lines[-1] or "Received face comparison request" in "".join(lines[-10:]):
                    print("SUCCESS: Log content verified.")
                else:
                    print("WARNING: Expected log message not found in recent logs.")
            else:
                print("WARNING: Log file is empty.")
    else:
        print(f"FAILURE: Log file not found at {log_file}")

if __name__ == "__main__":
    test_logging()
