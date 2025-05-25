# Plataforma LMS múltiple con SLAM visual

Este proyecto implementa una plataforma modular que permite experimentar con distintos algoritmos de SLAM (Simultaneous Localization and Mapping) utilizando videos monoculares como entrada. Está diseñado para ser interactivo, visualmente atractivo y extensible. También incluye funcionalidad para medir métricas de rendimiento como uso de CPU, memoria y tiempo de ejecución. 

Proyecto desarrollado por Jimena León Huertas, estudiante del Tecnológico de Costa Rica (TEC) durante el primer semestre de 2025, como parte del curso CE-1114 - Proyecto de Aplicación de la Ingeniería en Computadores. 

Se agradece el apoyo del profesor asesor MSc. Luis Alberto Chavarría Zamora.

## Propósito del proyecto

El proyecto fue creado con fines educativos e investigativos, como parte de un trabajo académico del Tecnológico de Costa Rica (TEC). Su objetivo es permitir a los estudiantes y desarrolladores:

- Probar diferentes variantes de algoritmos SLAM visuales.
- Visualizar trayectorias estimadas en 2D a partir de videos.
- Evaluar el consumo de recursos de cada implementación.
- Facilitar la integración y comparación de nuevas técnicas LMS.

## Características principales

- Interfaz gráfica desarrollada en `Tkinter` con diseño moderno y responsivo.
- Soporte para múltiples LMS almacenados en subcarpetas como `LMS_ORB_with_BA`, `LMS_visual`, etc.
- Ejecución de scripts LMS individuales con selección dinámica de video `.mp4`.
- Cálculo automático de métricas de rendimiento (`psutil`).
- Visualización automática de resultados y exportación en CSV y PNG.
- Resultados organizados por fecha y tipo de LMS.

## Requisitos del sistema

- Ubuntu 22.04 o superior.
- Python 3.12 (con soporte para venv y tkinter).
- Acceso a internet para instalación de dependencias.
- Video de entrada en formato `.mp4`.

## Instalación

1. Clone este repositorio o descargue los archivos en una carpeta local:

```bash
git clone https://github.com/menaleon/slam-2-diseno.git
cd slam-2-diseno
```
2. Genere el ambiente virtual e instale las dependencias con este script:

```bash
chmod +x setup.sh
./setup.sh
```

## Ejecución

1. Active el ambiente virtual de Python, el cual contiene lo necesario para la ejecución.

```bash
source venv/bin/activate
```

2. Ejecute el archivo de la interfaz.

```bash
python3 interfaz.py
```

## Manual de usuario

pendiente

### Resultados

Los resultados se almacenan automáticamente en la siguiente jerarquía de carpetas:

resultados/
└── LMS_ORB_with_BA/
└── 1245_2505_2025/
├── trayectoria_LMS_ORB_with_BA.csv
└── trayectoria_LMS_ORB_with_BA.png