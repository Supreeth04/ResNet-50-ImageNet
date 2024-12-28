#!/bin/bash

# Update package list
sudo apt-get update

# Install build essentials and pkg-config
sudo apt-get install -y build-essential pkg-config

# Install FFCV dependencies
sudo apt-get install -y \
    libturbojpeg-dev \
    libopencv-dev \
    python3-opencv \
    gcc \
    g++ \
    ninja-build

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch first (with CUDA 12.4 support)
#pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt

# Downgrade setuptools to a compatible version
pip install setuptools==65.5.1

# Clone and install FFCV from source
git clone https://github.com/libffcv/ffcv.git
cd ffcv
CFLAGS="-mavx2" pip install -e .
cd .. 