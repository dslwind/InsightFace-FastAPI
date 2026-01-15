FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Configure apt sources to use CERNET mirror
RUN echo "deb https://mirrors.cernet.edu.cn/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.cernet.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.cernet.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.cernet.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and configure global mirror
RUN python3 -m pip install -i https://mirrors.cernet.edu.cn/pypi/web/simple --no-cache-dir --upgrade pip && \
    pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
