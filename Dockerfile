# Base de Ubuntu 22.04.5 LTS
FROM ubuntu:22.04

# Configurar variables de entorno para evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHON_VERSION=3.10.12

# Actualizar sistema e instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    software-properties-common \
    sudo \
    cmake \
    build-essential \
    libglew-dev \
    libpython3.10-dev \
    python3.10 \
    python3.10-venv \
    python3.10-tk \
    python3-pip \
    git \
    pkg-config \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libwayland-dev \
    ffmpeg \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswscale-dev \
    libeigen3-dev \
    libdc1394-dev \
    libraw1394-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*  # Limpiar caché de APT para reducir el tamaño del contenedor

# Crear y activar un entorno virtual con Python 3.10
RUN python3.10 -m venv /venv310 && \
    /venv310/bin/pip install --upgrade pip

# Instalar dependencias dentro del entorno virtual
RUN /venv310/bin/pip install \
    pygame \
    numpy \
    ttkbootstrap \
    psutil \
    matplotlib \
    scikit-image \
    opencv-python opencv-contrib-python \
    PyOpenGL PyOpenGL_accelerate \
    g2o-python

# Definir el entorno virtual como predeterminado
ENV PATH="/venv310/bin:$PATH"

# Establecer el directorio de trabajo predeterminado
WORKDIR /workspace

# Copiar un script de instalación de Pangolin dentro del contenedor
COPY install_pangolin.sh /opt/install_pangolin.sh

# Comando predeterminado (bash interactivo)
CMD ["/bin/bash"]
