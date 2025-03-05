#!/bin/bash

# Instalar OpenCV y Open3D
pip install opencv-python open3d numpy scipy sklearn

# Clonar e instalar ORB-SLAM3 (requiere CMake y Pangolin)
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
./build.sh
