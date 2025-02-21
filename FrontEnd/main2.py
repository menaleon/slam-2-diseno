import os
import subprocess
import threading
import time
from slam_tests import Testing

# Lista de rutas de conjuntos de datos y sus algoritmos
datasets = [
    #{"path": "/home/faleivac/Documents/GitHub/TFG_FL_SLAM/Dataset/Estabilidad/ConEstab_monocular/prueba20", "algorithm": "monocular"},
    #{"path": "/home/faleivac/Documents/GitHub/TFG_FL_SLAM/Dataset/Estabilidad/ConEstab_monocular/prueba22", "algorithm": "sift"},
    #{"path": "/home/faleivac/Documents/GitHub/TFG_FL_SLAM/Dataset/Estabilidad/ConEstab_monocular/prueba20", "algorithm": "optimize"},
    {"path": "/home/faleivac/Documents/GitHub/TFG_FL_SLAM/Dataset/LIDAR/Inside/prueba14", "algorithm": "monocular"},
    #{"path": "/home/faleivac/Documents/GitHub/TFG_FL_SLAM/Dataset/LIDAR/Inside/prueba14", "algorithm": "sift"},
    #{"path": "/home/faleivac/Documents/GitHub/TFG_FL_SLAM/Dataset/LIDAR/Inside/prueba16", "algorithm": "optimize"},
]

def execute_algorithm(dataset_path, algorithm):
    testing = Testing(dataset_path, algorithm)
    try:
        def monitor_resources():
            while testing.end_time is None:
                testing.take_measure()
                time.sleep(1)

        # Iniciar el hilo para monitorear los recursos
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        # Selección de comandos de acuerdo al algoritmo
        if algorithm == "monocular":
            print("Comenzando monocular")
            subprocess.run(['python3', '../SLAMPy-Monocular-SLAM-implementation-in-python/slam.py', dataset_path], check=True)
        elif algorithm == "lidar":
            subprocess.run(['python3', '../StrayWithPangolin/main.py', dataset_path], check=True)
        elif algorithm == "sift":
            print("Comenzando sift")
            subprocess.run(['python3', '../SIFT_SLAM/slam.py', dataset_path], check=True)
        elif algorithm == "optimize":
            print("Comenzando optimize")
            subprocess.run(['python3', '../Bundle_Adjustment_SLAM/slam.py', dataset_path], check=True)
        else:
            print(f"Algoritmo desconocido: {algorithm}")
            return
        
        print("End data")

        # Finalizar y guardar el testing al terminar
        testing.end_testing()
        testing.save_data()
        print(f"{algorithm.capitalize()} SLAM en {dataset_path} completado exitosamente.")
    except subprocess.CalledProcessError as e:
        # Finalizar y guardar el testing antes de detener la ejecución
        testing.end_testing()
        testing.save_data()
        print(f"Error al ejecutar {algorithm} SLAM en {dataset_path}:\n{e}")
        raise

def main():
    for dataset in datasets:
        dataset_path = dataset["path"]
        algorithm = dataset["algorithm"]

        print(f"Iniciando {algorithm.capitalize()} SLAM en {dataset_path}...")

        # Verificar archivos necesarios según el tipo de SLAM
        if algorithm == "monocular":
            if not os.path.exists(os.path.join(dataset_path, 'rgb.mp4')):
                print(f"Error: La carpeta {dataset_path} no contiene 'rgb.mp4'.")
                break
        elif algorithm == "lidar":
            if not (os.path.exists(os.path.join(dataset_path, 'odometry.csv')) and 
                    os.path.exists(os.path.join(dataset_path, 'depth'))):
                print(f"Error: La carpeta {dataset_path} no contiene los archivos necesarios para SLAM Lidar.")
                break

        try:
            execute_algorithm(dataset_path, algorithm)
        except Exception as e:
            print(f"Deteniendo la ejecución debido a un error en {algorithm.capitalize()} SLAM: {e}")
            break

if __name__ == "__main__":
    main()
