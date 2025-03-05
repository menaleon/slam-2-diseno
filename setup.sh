#!/bin/bash

# Este script instala las dependencias necesarias en un ambiente virtual de Ubuntu24.04
# Instalar Python 3.10 y paquetes necesarios
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo apt-get install cmake build-essential libglew-dev libpython3.10-dev

# Generar un ambiente virtual donde iran las dependencias
python3.10 -m venv venv310
source venv310/bin/activate

# Instalacion de dependencias
pip install --upgrade pip
sudo apt install -y python3.10-tk       # tkinter
pip install pygame
pip install numpy
pip install ttkbootstrap
pip install psutil
pip install matplotlib
pip install scikit-image
pip install opencv-python opencv-contrib-python
pip install PyOpenGL PyOpenGL_accelerate
pip install g2o-python

# Instalar Pangolin
sudo apt update
sudo apt install build-essential cmake git pkg-config libgl1-mesa-dev libglew-dev \
                 libegl1-mesa-dev libwayland-dev ffmpeg libavcodec-dev libavutil-dev \
                 libavformat-dev libswscale-dev python3.10-dev
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python3.10) -DCMAKE_BUILD_TYPE=Release
cmake --build .            # Compilar Pangolin (todos sus componentes)
cmake --build . --target pypangolin_pip_install

# import pypangolin as pango
# print(pango.__version__)  # Opcional: verificar versión del módulo

echo "Todas las dependencias han sido instaladas correctamente en el entorno virtual 'env310'"
