#!/bin/bash

# Update package list
sudo apt-get update

# Install build essentials and pkg-config
sudo apt-get install -y build-essential pkg-config

# Install libturbojpeg
sudo apt-get install -y libturbojpeg-dev

# Install OpenCV dependencies
sudo apt-get install -y libopencv-dev python3-opencv

# Install Python requirements
pip install -r requirements.txt 