#!/bin/bash

set -e  # Detener la ejecución si hay un error

echo "Iniciando instalación de Pangolin..."

# Definir directorio de instalación
INSTALL_DIR="/opt/Pangolin"

# Clonar Pangolin solo si no existe
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Clonando Pangolin..."
    cd /opt
    git clone --recursive https://github.com/uoip/pangolin
else
    echo "Pangolin ya está clonado. Saltando esta etapa."
fi

# Entrar al directorio de Pangolin
cd "$INSTALL_DIR"

# Crear carpeta build si no existe
mkdir -p build
cd build

# Configurar la compilación
echo "Ejecutando CMake..."
cmake .. -DBUILD_PANGOLIN_FFMPEG=OFF -DPYBIND11_PYTHON_VERSION=3.10 -DCMAKE_BUILD_TYPE=Release

# Compilar
echo "Compilando Pangolin..."
make -j$(nproc)

# Instalar PyPangolin
echo "Instalando PyPangolin..."
cmake --build . --target pypangolin_pip_install

echo "Pangolin ha sido instalado correctamente."
