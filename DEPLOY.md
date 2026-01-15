# Deployment Guide

This guide describes how to deploy the InsightFace-FastAPI application using Docker with GPU support.

## Prerequisites

1.  **NVIDIA Drivers**: Ensure appropriate NVIDIA drivers are installed on your host machine.
2.  **Docker**: Install Docker Engine.
3.  **NVIDIA Container Toolkit**: Install and configure the NVIDIA Container Toolkit to allow Docker to access your GPU.
    *   Follow the official guide: [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Deployment Steps

### 1. Clone the repository (if not already done)

```bash
git clone <your-repo-url>
cd insightface-FastAPI
```

### 2. Configure Environment Variables

Ensure you have a `.env` file in the root directory. You can start by copying the example:

```bash
cp .env.example .env
```
Edit `.env` to match your desired configuration.

### 3. Build and Run with Docker Compose (Recommended)

This method automates building the image and running the container with the correct configurations.

```bash
docker-compose up -d --build
```

- `-d`: Run in detached mode (in the background).
- `--build`: Force a rebuild of the image.

### 4. Verify Deployment

Check if the container is running:

```bash
docker-compose ps
```

View logs to ensure the application started and models are loaded:

```bash
docker-compose logs -f
```

Access the API welcome message at: `http://localhost:8000/`

### 5. Stop the Application

```bash
docker-compose down
```

## Running with Docker (Manual)

If you prefer `docker run` without Compose:

1.  **Build the image:**
    ```bash
    docker build -t insightface-fastapi .
    ```

2.  **Run the container:**
    ```bash
    docker run -d \
      --name insightface-fastapi \
      --gpus all \
      -p 8000:8000 \
      -v $(pwd)/models:/app/models \
      -v $(pwd)/logs:/app/logs \
      -v $(pwd)/.env:/app/.env \
      insightface-fastapi
    ```

## Troubleshooting

-   **"could not select device driver"**: Ensure NVIDIA Container Toolkit is installed and you've restarted the Docker daemon (`sudo systemctl restart docker`).
-   **Model loading errors**: Check `logs/` or `docker logs` for network issues if models are being downloaded, or verify model paths in `.env` if using local models.
