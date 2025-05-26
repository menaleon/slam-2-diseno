import csv
from datetime import datetime
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

class PoseGraphSLAM:
    def __init__(self, fx=700, fy=700, cx=320, cy=240):
        """
        Inicializa los componentes del sistema SLAM basado en grafo de poses.
        Configura el detector de caracteristicas, la matriz de la camara y las variables internas.

        :param name: Nombre base para los archivos de salida.
        """

        # Parametros de la camara para la matriz intrinseca
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Nombre del tipo de SLAM     
        self.name = Path(__file__).resolve().parent.name

        # Detector ORB con hasta 2000 keypoints por frame
        self.orb_detector = cv2.ORB_create(2000)

        # Matcher de fuerza bruta usando distancia Hamming (para descriptores binarios)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Matriz intrinseca de la camara (fx, fy, cx, cy)
        self.camera_matrix = np.array([
                                    [self.fx,    0, self.cx],
                                    [0,    self.fy, self.cy],
                                    [0,        0,      1]
                                ])

        # Lista de poses absolutas de cada keyframe (matrices 4x4)
        self.keyframe_poses = []

        # Transformaciones relativas entre keyframes consecutivos
        self.relative_transformations = []

        # Atributos del keyframe anterior
        self.previous_keyframe_image = None
        self.previous_keyframe_keypoints = None
        self.previous_keyframe_descriptors = None
        self.previous_keyframe_pose = np.eye(4)

        # Control de espaciado entre keyframes
        self.frame_counter = 0
        self.min_frame_gap = 5
        self.min_keyframe_translation = 0.05

    def filter_matches_lowe_ratio(self, descriptors1, descriptors2, ratio=0.75):
        """
        Aplica el criterio de Lowe para filtrar coincidencias ambiguas entre descriptores.

        :param descriptors1: Descriptores del primer conjunto.
        :param descriptors2: Descriptores del segundo conjunto.
        :param ratio: Umbral para la prueba de razon.
        :return: Lista de coincidencias filtradas.
        """
        knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches

    def process_frame(self, frame):
        """
        Procesa un frame del video, detectando keypoints y descriptores.
        Estima el movimiento relativo si hay keyframe anterior y decide si se guarda un nuevo keyframe.

        :param frame: Imagen del frame actual en formato BGR.
        """
        # Convierte el frame a escala de grises
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta puntos clave y descriptores ORB
        keypoints, descriptors = self.orb_detector.detectAndCompute(grayscale_frame, None)

        # Si hay un keyframe previo con descriptores, procede a estimar movimiento relativo
        if self.previous_keyframe_descriptors is not None and descriptors is not None and len(keypoints) > 0:
            # Coincidencias con la prueba de Lowe
            matches = self.filter_matches_lowe_ratio(self.previous_keyframe_descriptors, descriptors)

            if len(matches) > 30:
                # Extrae las coordenadas de puntos emparejados
                points_prev = np.float32([self.previous_keyframe_keypoints[m.queryIdx].pt for m in matches])
                points_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

                # Estima la matriz esencial entre las dos vistas
                essential_matrix, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix, method=cv2.RANSAC, threshold=1.0)

                if essential_matrix is not None:
                    # Recupera la rotacion y traslacion entre keyframes
                    _, rotation, translation, _ = cv2.recoverPose(essential_matrix, points_prev, points_curr, self.camera_matrix)

                    # Construye la matriz de transformacion relativa 4x4
                    relative_pose = np.eye(4)
                    relative_pose[:3, :3] = rotation
                    relative_pose[:3, 3] = translation.ravel()

                    # Calcula la nueva pose global acumulando la transformacion relativa
                    current_pose = self.previous_keyframe_pose @ relative_pose

                    # Calcula magnitud de traslacion para determinar si se agrega nuevo keyframe
                    translation_magnitude = np.linalg.norm(relative_pose[:3, 3])

                    # Condiciones para insertar un nuevo keyframe
                    if self.frame_counter >= self.min_frame_gap or translation_magnitude > self.min_keyframe_translation:
                        self.keyframe_poses.append(current_pose.copy())
                        self.relative_transformations.append(relative_pose.copy())
                        self.previous_keyframe_keypoints = keypoints
                        self.previous_keyframe_descriptors = descriptors
                        self.previous_keyframe_pose = current_pose
                        self.frame_counter = 0
                    else:
                        self.frame_counter += 1
        else:
            # Si no hay keyframe previo, este se considera el primero
            self.keyframe_poses.append(np.eye(4))
            self.previous_keyframe_keypoints = keypoints
            self.previous_keyframe_descriptors = descriptors
            self.previous_keyframe_pose = np.eye(4)

    def optimize_pose_graph(self):
        """
        Optimiza globalmente las posiciones de los keyframes usando las transformaciones relativas acumuladas.

        :return: Trayectoria optimizada en el plano X-Z.
        """
        # Extrae las posiciones absolutas de los keyframes (X, Y, Z)
        keyframe_positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])

        # Vector inicial de parametros (todas las posiciones concatenadas)
        initial_parameters = keyframe_positions.flatten()

        # Define la funcion de error basada en diferencias entre movimientos estimados y observados
        def residual_function(parameters):
            residuals = []
            optimized_positions = parameters.reshape((-1, 3))
            for i in range(1, len(optimized_positions)):
                predicted_motion = optimized_positions[i] - optimized_positions[i - 1]
                measured_motion = self.relative_transformations[i - 1][:3, 3]
                residuals.extend((predicted_motion - measured_motion).tolist())
            return residuals

        # Ejecuta la optimizacion no lineal por minimos cuadrados
        result = least_squares(residual_function, initial_parameters, verbose=0)

        # Reconstruye la trayectoria optimizada
        optimized_positions = result.x.reshape((-1, 3))
        return optimized_positions[:, [0, 2]]  # Devuelve solo X-Z

    def save_trajectory_outputs(self, trajectory, input_video_path):
        """
        Guarda la trayectoria optimizada en formato CSV y genera una visualizacion en PNG.

        :param trajectory: Matriz Nx2 con las coordenadas X-Z de la trayectoria.
        """
        # === Preparar ruta de salida con timestamp ===
        tipo_lms = self.name  # por ejemplo: "LMS_ORB_with_BA"
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y") 
        output_dir = os.path.join("resultados", tipo_lms, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, f"trayectoria_{self.name}") # Nombre base de salida

        # Guarda los datos de la trayectoria en archivo CSV
        with open(output_base + ".csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Z"])
            writer.writerows(trajectory)

        # Visualiza la trayectoria optimizada
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Optimized Trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color='g', label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='r', label='End')
        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.title(f"{self.name} para {Path(input_video_path).name}")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.savefig(output_base + ".png")
        plt.close()

    def process_video_input(self, video_path):
        """
        Procesa un video completo frame por frame, construyendo el grafo de poses y optimizandolo al final.

        :param video_path: Ruta al archivo de video de entrada.
        """
        # Abre el video de entrada
        video_capture = cv2.VideoCapture(video_path)

        # Procesa cada frame hasta que se termine el video
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
            self.process_frame(frame)

        # Libera recursos del video
        video_capture.release()

        # Ejecuta la optimizacion y guarda resultados
        optimized_trajectory = self.optimize_pose_graph()
        self.save_trajectory_outputs(optimized_trajectory, input_video_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 main.py <ruta_al_video>")
        sys.exit(1)

    input_video_path = sys.argv[1]

    if not os.path.exists(input_video_path):
        print(f"Error: no se encontró el archivo {input_video_path}")
        sys.exit(1)

    # Inicializa el sistema SLAM
    slam_system = PoseGraphSLAM()

    # Ejecuta el procesamiento
    slam_system.process_video_input(input_video_path)

    print("Optimizacion global completada y trayectoria guardada.")
