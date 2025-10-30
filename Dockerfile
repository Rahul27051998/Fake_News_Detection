# Dockerfile
# Use official PyTorch image with CUDA if you have GPU. Change tag to cpu-only if needed.
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# Install system deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Expose port for API
EXPOSE 8000

# Entrypoint: choose RUN_MODE (train or serve) via container cmd/override
# Default runs the FastAPI server
CMD ["bash", "/app/scripts/entrypoint.sh", "serve"]
