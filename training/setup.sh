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
    ninja-build \
    numactl \
    python3-dev

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (optimized for A10G)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Clean any existing ffcv installations
pip uninstall -y ffcv

# Install basic requirements (except ffcv)
pip install -r <(grep -v ffcv requirements.txt)

# Clone and install FFCV with proper configuration
rm -rf ffcv
git clone https://github.com/libffcv/ffcv.git
cd ffcv
# Install in development mode with specific compiler flags
CFLAGS="-mavx2 -O3" pip install -e . --no-cache-dir
cd ..

# Verify FFCV installation
python -c "import ffcv; print(f'FFCV version: {ffcv.__version__}')"

# Install NVIDIA Apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..

# Set up system optimizations
echo "
# Optimize system for ML workload
vm.swappiness=10
vm.dirty_background_ratio=5
vm.dirty_ratio=80
net.core.rmem_max=67108864
net.core.wmem_max=67108864
net.ipv4.tcp_rmem=4096 87380 67108864
net.ipv4.tcp_wmem=4096 65536 67108864
net.core.netdev_max_backlog=30000
net.core.somaxconn=1024" | sudo tee -a /etc/sysctl.conf

# Apply system optimizations
sudo sysctl -p

# Set GPU persistence mode for better performance
sudo nvidia-smi -pm 1

# Set GPU power limit to maximum
sudo nvidia-smi -pl 250

# Configure GPU clocks for optimal performance
sudo nvidia-smi -ac 1215,1410

# Print installation status
echo "Installation completed. Checking environment..."
python -c "import torch; import ffcv; import apex; print('All required packages imported successfully')" 