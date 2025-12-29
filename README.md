# InsightFace FastAPI Service

A high-performance face comparison API built with [FastAPI](https://fastapi.tiangolo.com/) and [InsightFace](https://github.com/deepinsight/insightface). This service allows you to compare faces from URLs or Base64 strings with advanced filtering and detailed metadata.

## Features

- **Face Comparison**: Compare two faces and get high-accuracy similarity scores.
- **Flexible Inputs**: Supports Image URLs and Base64 encoded strings.
- **Advanced Filtering**: Filter faces by minimum size, detection confidence, and limit the number of faces processed.
- **Rich Metadata**: Returns processing time, face counts, and input parameters in every response.
- **Configuration Management**: Fully configurable via environment variables (`.env`).
- **Robust Logging**: Detailed logging to both console and rotating files.
- **High Performance**: Optimized with ONNX Runtime and optional CUDA (GPU) support.

## Prerequisites

- **Python**: 3.8+ (Tested with Python 3.12 and NumPy < 2)
- **CUDA/cuDNN**: Optional, for GPU acceleration (set `ENABLE_CUDA=True` in `.env`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd insightface-FastAPI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the environment:**
    Copy the example environment file and edit it:
    ```bash
    cp .env.example .env
    ```

## Configuration

The application can be configured via environment variables or a `.env` file. Key settings include:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `INSIGHTFACE_MODEL_NAME` | The InsightFace model pack to use. | `buffalo_l` |
| `ENABLE_CUDA` | Enable GPU acceleration (requires CUDA). | `False` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR). | `INFO` |
| `DEFAULT_THRESHOLD` | Default similarity threshold for a match. | `0.3` |

Refer to `.env.example` for a full list of available settings.

## Running the Application

Start the server using `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### `POST /face/compare`

Compare two faces. This endpoint accepts a JSON body.

**Request Body (`application/json`):**
```json
{
  "image1": "https://example.com/face1.jpg",
  "image2": "data:image/jpeg;base64,...",
  "threshold": 0.3,
  "limit_faces": 0,
  "min_face_size": 0,
  "best_face_strategy": "center",
  "compare_all_faces": false
}
```

**Response:**
```json
{
  "is_same_person": true,
  "similarity_score": 0.8542,
  "status": "success",
  "face_counts": {
    "image1": 1,
    "image2": 1
  },
  "processing_time_ms": 125.5,
  "parameters": {
    "threshold": 0.3,
    "detection_threshold": 0.6,
    "limit_faces": 0,
    "min_face_size": 0,
    "best_face_strategy": "center",
    "input_format": "auto"
  }
}
```

## Logging

Logs are written to:
1.  **Console**: Real-time output for monitoring.
2.  **File**: `logs/insightface_api.log` with automatic rotation (10MB size, 5 backups).

The log level can be adjusted via the `LOG_LEVEL` setting in your `.env` file.

## Testing

Comprehensive test scripts are located in the `tests/` directory:

- `test_advanced_api.py`: Tests filtering, limiting, and structure.
- `test_all_faces.py`: Tests multi-face comparison logic.
- `test_flexible_inputs.py`: Verifies URL and Base64 handling.
- `test_strategies.py`: Tests different face selection strategies.

Run a test with:
```bash
python tests/test_advanced_api.py
```

## Project Structure

```text
.
├── app/
│   ├── main.py            # Application entry point & lifespan management
│   ├── config.py          # Configuration management (Pydantic Settings)
│   ├── logger.py          # Logging initialization
│   ├── routers/           # API routes
│   ├── services/          # Core logic (FaceService)
│   └── schemas.py         # Request/Response models
├── tests/                 # Comprehensive test suite
├── logs/                  # Log files (auto-generated)
├── requirements.txt       # Project dependencies
├── .env.example           # Template for configuration
└── README.md              # Project documentation
```
