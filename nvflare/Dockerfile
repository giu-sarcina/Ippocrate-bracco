# Base image with CUDA 12.4 and PyTorch preinstalled
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Switch to root to install system packages
USER root

# Install network diagnostic tools
RUN apt-get update && apt-get install -y \
    iputils-ping \
    net-tools \
    dnsutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory     
WORKDIR /workspace

# Copy requirements first to take advantage of Docker layer caching
COPY requirements.txt .

# Upgrade pip and setuptools, then install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools \
    && python3 -m pip install -r requirements.txt

# Install NVFLARE separately 
RUN python3 -m pip install nvflare==2.5.2

# Default working directory inside the container
WORKDIR /workspace
