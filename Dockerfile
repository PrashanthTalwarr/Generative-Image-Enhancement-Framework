# Medical Image Enhancement GAN - Production Dockerfile
# Optimized for AWS deployment with GPU support

FROM tensorflow/tensorflow:2.13.0-gpu

LABEL maintainer="Medical GAN Team"
LABEL description="Medical Image Enhancement GAN API"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV FLASK_APP=src/api/flask_app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Image processing libraries
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    # Medical imaging support
    dcm2niix \
    # System utilities
    curl \
    wget \
    git \
    htop \
    vim \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p \
    data/synthetic \
    data/raw \
    data/processed \
    models/checkpoints \
    models/final \
    results \
    logs \
    temp \
    && chmod -R 755 /app

# Create non-root user for security
RUN groupadd -r medgan && \
    useradd -r -g medgan -d /app -s /bin/bash medgan && \
    chown -R medgan:medgan /app

# Install additional medical imaging packages (optional)
RUN pip install --no-cache-dir \
    pydicom \
    nibabel \
    SimpleITK \
    || echo "Optional medical packages installation failed, continuing..."

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose API port
EXPOSE 5000

# Switch to non-root user
USER medgan

# Default command - can be overridden
CMD ["python", "src/api/flask_app.py"]

# Alternative commands for different use cases:
# Training: docker run medical-gan python train.py
# API: docker run -p 5000:5000 medical-gan
# Shell: docker run -it medical-gan bash

# Build instructions:
# docker build -t medical-gan:latest .
# docker run -p 5000:5000 -v $(pwd)/models:/app/models medical-gan:latest

# For GPU support:
# docker run --gpus all -p 5000:5000 medical-gan:latest

# For production with mounted volumes:
# docker run -d \
#   --name medical-gan-api \
#   --gpus all \
#   -p 5000:5000 \
#   -v /path/to/models:/app/models \
#   -v /path/to/data:/app/data \
#   -v /path/to/logs:/app/logs \
#   -e FLASK_ENV=production \
#   medical-gan:latest