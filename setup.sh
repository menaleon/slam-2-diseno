#!/bin/bash

# Exit immediately if an error occurs
set -e

echo "Installing system dependencies..."

# Install required system dependencies
sudo apt update
sudo apt install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    python3.12-tk \
    libgl1 \
    libglib2.0-0

echo "System dependencies installed."

# Create virtual environment with Python 3.12
echo "Creating virtual environment..."
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install \
    opencv-python \
    opencv-contrib-python \
    numpy \
    matplotlib \
    scipy \
    psutil \
    pillow

echo "Environment successfully configured."
