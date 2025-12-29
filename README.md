# InsightFace FastAPI Service

A high-performance face comparison API built with [FastAPI](https://fastapi.tiangolo.com/) and [InsightFace](https://github.com/deepinsight/insightface). This service allows you to upload two images and determine if they represent the same person by calculating the cosine similarity between their face embeddings.

## Features

- **Face Comparison**: Compare two face images and get a similarity score.
- **FastAPI Powered**: High performance, easy to use, and automatic Swagger documentation.
- **InsightFace Integration**: Utilizes state-of-the-art face recognition models (Buffalo_L) and ONNX Runtime (GPU recommended).

## Prerequisites

- **Python**: 3.8+ (Tested with NumPy < 2 as per InsightFace requirements)
- **CUDA/cuDNN**: If using `onnxruntime-gpu` (recommended for performance).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd insightface-FastAPI
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/WSL
    # or
    venv\Scripts\activate     # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `onnxruntime-gpu` is included in `requirements.txt`. If you do not have a GPU, run `pip uninstall onnxruntime-gpu && pip install onnxruntime`.*

## Running the Application

Start the development server using `uvicorn`:

```bash
uvicorn app.main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

## API Documentation

Once the server is running, you can access the interactive API documentation (Swagger UI) at:
*   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoints

#### `POST /face/compare`

Compare two images containing faces.

**Request:**
- `file1`: Image file (JPEG/PNG)
- `file2`: Image file (JPEG/PNG)

**Response:**
```json
{
  "similarity": 0.85,
  "match": true
}
```

## Testing

A test script is provided in the `tests/` directory to verify the API functionality.

1.  **Ensure the server is running** (see "Running the Application" above).
2.  **Run the test script:**

    ```bash
    python tests/test_api.py
    ```

    The script uses `face1.png` and `face2.png` located in the `tests/` directory to simulate a request.

## Project Structure

```text
.
├── app/
│   ├── main.py            # Application entry point
│   ├── routers/           # API routes definition
│   ├── services/          # Business logic (InsightFace handling)
│   └── schemas.py         # Pydantic models
├── tests/
│   ├── test_api.py        # Test script
│   ├── face1.png          # Test image 1
│   └── face2.png          # Test image 2
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```
