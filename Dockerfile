# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    nano \
    wget \
    curl \
    unzip \
    htop \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip setuptools wheel

# Set environment variables for CUDA and PyTorch
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST=8.0
ENV TORCH_USE_CUDA_DSA=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1
ENV CUDA_LAUNCH_BLOCKING=0

# Install blinker first
RUN pip install --no-cache-dir blinker==1.9

# Install bitsandbytes and core dependencies
RUN pip install --no-cache-dir --ignore-installed bitsandbytes && \
    pip install --no-cache-dir --ignore-installed \
    torch==2.1.0 \
    transformers==4.36.2 \
    datasets==2.16.1 \
    accelerate==0.25.0 \
    scikit-learn==1.3.0 \
    pandas==2.1.4 \
    numpy==1.24.3 \
    tqdm==4.66.1 \
    psutil==5.9.6

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/configs

# Default command to run trainer
CMD ["python", "src/auto_trainer.py"] 