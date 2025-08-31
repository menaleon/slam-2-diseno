# Multi-SLAM Platform with Visual SLAM

This project implements a modular platform that allows experimenting with different SLAM (Simultaneous Localization and Mapping) algorithms using monocular videos as input. It is designed to be interactive, visually appealing, and extensible. It also includes functionality to measure performance metrics such as CPU usage, memory, and execution time.  

Project developed by Jimena León Huertas, student at the Tecnológico de Costa Rica (TEC) during the first semester of 2025, as part of the course CE-1114 – Computer Engineering Application Project.  

Special thanks to the advisor MSc. Luis Alberto Chavarría Zamora.  

## Project Purpose

The project was created for educational and research purposes, as part of an academic work at Tecnológico de Costa Rica (TEC). Its main goal is to enable students and developers to:

- Test different variants of visual SLAM algorithms.  
- Visualize estimated 2D trajectories from videos.  
- Evaluate the resource usage of each implementation.  
- Facilitate integration and comparison of new SLAM techniques.  

## Main Features

- Graphical interface developed in `Tkinter` with a modern and responsive design.  
- Support for multiple SLAM implementations stored in subfolders such as `SLAM_ORB_with_PG`, `SLAM_visual`, etc.  
- Execution of individual SLAM scripts with dynamic `.mp4` video selection.  
- Automatic performance metrics calculation (`psutil`).  
- Automatic result visualization and export in CSV and PNG.  
- Results organized by date and SLAM type.  

## System Requirements

- Ubuntu 22.04 or newer.  
- Python 3.12 (with venv and tkinter support).  
- Internet access for dependency installation.  
- Input video in `.mp4` format.  

## Installation

1. Clone this repository or download the files into a local folder:

```bash
git clone https://github.com/menaleon/slam-2-diseno.git
cd slam-2-diseno
```

2. Generate the virtual environment and install dependencies with the provided script:

```bash
chmod +x setup.sh
./setup.sh
```

## Execution

1. Activate the Python virtual environment (includes all required packages):

```bash
source venv/bin/activate
```

2. Run the UI script:

```bash
python3 ui.py
```

## User Manual

Pending  

### Results  

The results are automatically stored in a folder hierarchy like the following:

```
results/
    └── SLAM_ORB_with_PG/
        └── 1245_2505_2025/             (hour-minute_day-month_year)
            ├── trajectory_SLAM_ORB_with_PG.csv
            └── trajectory_SLAM_ORB_with_PG.png
    └── SLAM_visual
        └── 1304_2505_2025/             
            ├── trajectory_SLAM_visual.csv
            └── trajectory_SLAM_visual.png
```