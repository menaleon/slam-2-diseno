#!/bin/bash

# Crear entorno virtual en Python 3.12
python3.12 -m venv venv

# Activar el entorno virtual
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install opencv-python numpy matplotlib opencv-contrib-python scipy psutil
