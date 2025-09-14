# Mario RL Docker Container
# Multi-stage build for optimized container size

# Build argument for Python version
ARG PYTHON_VERSION=3.10

# Base stage with system dependencies
FROM python:${PYTHON_VERSION}-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Graphics and video
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    freeglut3-dev \
    # X11 for GUI (optional)
    xvfb \
    x11-utils \
    # Audio libraries for NES emulation
    libasound2-dev \
    libpulse-dev \
    # Additional OpenCV dependencies
    libglib2.0-0 \
    libgtk-3-0 \
    # FFmpeg for video processing
    ffmpeg \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directory
WORKDIR /workspace

# Runtime stage for production use
FROM base AS runtime

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mario && \
    chown -R mario:mario /workspace

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies as root
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p /workspace/models \
             /workspace/videos \
             /workspace/logs \
             /workspace/data && \
    chown -R mario:mario /workspace

# Switch to non-root user
USER mario

# Set environment variables for X11 forwarding
ENV DISPLAY=:99
ENV QT_X11_NO_MITSHM=1

# Expose port for TensorBoard
EXPOSE 6006

# Default command
CMD ["python", "training_scripts/train_mario.py"]

# Development stage with additional tools
FROM runtime AS development

# Switch back to root for installing dev tools
USER root

# Install development dependencies globally
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    matplotlib \
    seaborn \
    tensorboard \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Expose additional ports for development
EXPOSE 8888 6006

# Switch back to non-root user
USER mario

# Add jupyter to PATH for mario user
ENV PATH="/usr/local/bin:$PATH"

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]