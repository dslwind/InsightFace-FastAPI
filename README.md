# InsightFace FastAPI Service

A FastAPI service for face comparison built on top of [InsightFace](https://github.com/deepinsight/insightface).
It accepts image URLs or Base64 strings, detects faces, and returns similarity scores plus request metadata.

## Features

- Face detection with face count, bounding boxes, area, center points, and confidence scores
- Face comparison with InsightFace embeddings
- Image input via URL or Base64
- Multi-face filtering by size, confidence, and face limit
- Optional all-vs-all comparison for multi-face images
- Optional image rotation retries when detection fails
- Configurable logging and optional database request logging

## Prerequisites

- Python 3.9+
- CUDA/cuDNN is optional
- For GPU deployments, make sure ONNX Runtime GPU and your NVIDIA stack are compatible

## Installation

```bash
git clone https://github.com/dslwind/InsightFace-FastAPI.git
cd InsightFace-FastAPI
pip install -r requirements.txt
cp .env.example .env
```

## Configuration

The application is configured through environment variables or `.env`.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `INSIGHTFACE_MODEL_NAME` | InsightFace model pack name | `buffalo_l` |
| `INSIGHTFACE_ROOT` | Model/cache directory | `models` |
| `ENABLE_CUDA` | Enable CUDA provider | `False` |
| `ENABLE_TENSORRT` | Enable TensorRT provider | `False` |
| `DEFAULT_THRESHOLD` | Similarity threshold for a match | `0.3` |
| `DEFAULT_DETECTION_THRESHOLD` | Minimum face detection confidence | `0.6` |
| `DEFAULT_LIMIT_FACES` | Max faces per image, `0` means unlimited | `0` |
| `DEFAULT_MIN_FACE_SIZE` | Minimum face box size in pixels | `0` |
| `DEFAULT_BEST_FACE_STRATEGY` | `area`, `center`, `confidence`, or `score` | `center` |
| `DEFAULT_COMPARE_ALL_FACES` | Compare all detected faces | `False` |
| `DEFAULT_ENABLE_ROTATION` | Retry image1 using 90° rotations | `False` |
| `ENABLE_DB_LOGGING` | Persist request/response logs to DB | `False` |

## Running the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The model is initialized during application startup and warmed up once to reduce first-request latency.

## API Usage

### `POST /face/detect`

**Request body (`application/json`)**

```json
{
  "image": "https://example.com/group-photo.jpg",
  "limit_faces": 0,
  "min_face_size": 0,
  "detection_threshold": 0.6,
  "best_face_strategy": "center"
}
```

**Successful response**

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "face_count": 2,
    "image_size": {
      "width": 1280,
      "height": 720
    },
    "faces": [
      {
        "index": 1,
        "confidence": 0.9987,
        "bbox": {
          "x1": 120.5,
          "y1": 80.0,
          "x2": 260.5,
          "y2": 250.0,
          "width": 140.0,
          "height": 170.0,
          "area": 23800.0
        },
        "center": {
          "x": 190.5,
          "y": 165.0
        },
        "area": 23800.0
      }
    ],
    "processing_time_ms": 48.3,
    "parameters": {
      "detection_threshold": 0.6,
      "limit_faces": 0,
      "min_face_size": 0,
      "best_face_strategy": "center"
    }
  }
}
```

> `face_count` is the total number of faces found after filtering. The `faces` array may be truncated by `limit_faces`.

### `POST /face/compare`

**Request body (`application/json`)**

```json
{
  "image1": "https://example.com/face1.jpg",
  "image2": "data:image/jpeg;base64,...",
  "threshold": 0.3,
  "limit_faces": 0,
  "min_face_size": 0,
  "detection_threshold": 0.6,
  "best_face_strategy": "center",
  "compare_all_faces": false,
  "enable_rotation": false
}
```

**Successful response**

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "is_same_person": true,
    "similarity_score": 0.8542,
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
      "compare_all_faces": false,
      "enable_rotation": false
    }
  }
}
```

**Error response example**

```json
{
  "code": -1,
  "msg": "No face detected in image2",
  "data": {
    "is_same_person": false,
    "similarity_score": -1.0,
    "face_counts": {
      "image1": 0,
      "image2": 0
    },
    "processing_time_ms": 32.7,
    "parameters": {
      "threshold": 0.3,
      "detection_threshold": 0.6,
      "limit_faces": 0,
      "min_face_size": 0,
      "best_face_strategy": "center",
      "compare_all_faces": false,
      "enable_rotation": false
    }
  }
}
```

## Logging

Logs are written to:

1. Console
2. `logs/insightface_api.log` with rotation support

If `ENABLE_DB_LOGGING=True`, request and response data can also be stored in PostgreSQL or SQLite.

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── logger.py
│   ├── routers/
│   ├── services/
│   ├── schemas.py
│   └── database_logger.py
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```
