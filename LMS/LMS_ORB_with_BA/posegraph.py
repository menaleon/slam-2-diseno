import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import csv

class PoseGraphSLAM:
    def __init__(self, name="pose_graph_slam"):
        # Inicializa el detector ORB y el matcher de fuerza bruta
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Matriz intrinseca de la camara
        self.K = np.array([[700, 0, 320],
                           [0, 700, 240],
                           [0,   0,   1]])

        self.name = name
        self.keyframes = []      # Lista de poses absolutas (matrices 4x4)
        self.rel_poses = []      # Lista de transformaciones relativas entre keyframes
        self.prev_kf_img = None  # Imagen del keyframe anterior
        self.prev_kf_kp = None   # Keypoints del keyframe anterior
        self.prev_kf_des = None  # Descriptores del keyframe anterior
        self.prev_kf_pose = np.eye(4)  # Pose absoluta del keyframe anterior
        self.frame_count = 0     # Contador de frames para controlar el espaciado entre keyframes
        self.min_gap = 5         # Numero minimo de frames entre keyframes
        self.min_translation = 0.05  # Umbral minimo de traslacion para insertar nuevo keyframe

    def lowe_ratio_match(self, des1, des2, ratio=0.75):
        # Aplica la prueba de Lowe para filtrar coincidencias ambiguas
        knn_matches = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    def process_frame(self, frame):
        # Convierte el frame a escala de grises y extrae keypoints y descriptores
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_kf_des is not None and des is not None and len(kp) > 0:
            # Coincidencias con el keyframe anterior usando la prueba de Lowe
            matches = self.lowe_ratio_match(self.prev_kf_des, des)
            if len(matches) > 30:
                # Extrae puntos correspondientes en ambos frames
                pts1 = np.float32([self.prev_kf_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                # Estima la matriz esencial y recupera la pose relativa (R, t)
                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
                    rel_pose = np.eye(4)
                    rel_pose[:3, :3] = R
                    rel_pose[:3, 3] = t.ravel()

                    # Actualiza la pose global del nuevo keyframe
                    new_pose = self.prev_kf_pose @ rel_pose

                    # Comprueba si se debe insertar un nuevo keyframe
                    delta = np.linalg.norm(rel_pose[:3, 3])
                    if self.frame_count >= self.min_gap or delta > self.min_translation:
                        self.keyframes.append(new_pose.copy())
                        self.rel_poses.append(rel_pose.copy())
                        self.prev_kf_kp = kp
                        self.prev_kf_des = des
                        self.prev_kf_pose = new_pose
                        self.frame_count = 0
                    else:
                        self.frame_count += 1
        else:
            # Primer frame o sin descriptores: inicializa primer keyframe
            self.keyframes.append(np.eye(4))
            self.prev_kf_kp = kp
            self.prev_kf_des = des
            self.prev_kf_pose = np.eye(4)

    def pose_graph_optimization(self):
        # Extrae las posiciones (X, Y, Z) de los keyframes
        poses = np.array([kf[:3, 3] for kf in self.keyframes])
        x0 = poses.flatten()

        # Define la funcion de costo basada en diferencias entre poses sucesivas y transformaciones relativas
        def residuals(x):
            res = []
            poses_opt = x.reshape((-1, 3))
            for i in range(1, len(poses_opt)):
                pred = poses_opt[i] - poses_opt[i-1]  # Movimiento estimado entre poses
                rel = self.rel_poses[i-1][:3, 3]       # Movimiento observado entre poses
                res.extend((pred - rel).tolist())      # Diferencia como residual
            return res

        # Optimiza usando minimos cuadrados no lineales
        result = least_squares(residuals, x0, verbose=0)
        optimized_positions = result.x.reshape((-1, 3))
        return optimized_positions[:, [0, 2]]  # Retorna solo X y Z para visualizacion

    def save_outputs(self, trajectory):
        # Guarda la trayectoria optimizada en un archivo CSV
        base = f"trayectoria_{self.name}"
        with open(base + ".csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Z"])
            writer.writerows(trajectory)

        # Genera grafico de la trayectoria
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Optimized Trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color='g', label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='r', label='End')
        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.title(f"Trajectory 2D - {self.name}")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.savefig(base + ".png")
        plt.close()

    def process_video(self, video_path):
        # Procesa frame a frame el video de entrada
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

        # Aplica optimizacion global y guarda resultados
        optimized_trajectory = self.pose_graph_optimization()
        self.save_outputs(optimized_trajectory)

# ---------- Ejecucion del script ----------
if __name__ == "__main__":
    video_path = "/home/jimena/Escritorio/PROYECTO/monocular/aparta2.mp4"
    slam = PoseGraphSLAM()
    slam.process_video(video_path)
    print("Optimizacion global completada y trayectoria guardada.")
