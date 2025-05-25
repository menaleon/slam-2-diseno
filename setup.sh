#!/bin/bash

# Terminar inmediatamente si ocurre un error
set -e

echo "Instalando dependencias del sistema..."

# Instalar dependencias necesarias del sistema
sudo apt update
sudo apt install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    python3.12-tk \
    libgl1 \
    libglib2.0-0

echo "Dependencias del sistema instaladas."

# Crear entorno virtual con Python 3.12
echo "Creando entorno virtual..."
python3.12 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias de Python
echo "Instalando dependencias de Python..."
pip install \
    opencv-python \
    opencv-contrib-python \
    numpy \
    matplotlib \
    scipy \
    psutil \
    pillow

echo "Entorno configurado correctamente."
