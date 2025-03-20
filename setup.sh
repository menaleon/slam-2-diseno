#!/bin/bash

# Crear entorno virtual en Python 3.12
python3.12 -m venv project_env

# Activar el entorno virtual
source project_env/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install opencv-python numpy matplotlib opencv-contrib-python scipy open3d
