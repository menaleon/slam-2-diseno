import csv
from datetime import datetime
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors

class ICPVisualSLAM:
    def __init__(self):
        self.name = Path(__file__).resolve().parent.name

        self.keyframe_poses = []
        self.relative_transformations = []

        self.previous_edges = None
        self.previous_pose = np.eye(3)  # Pose en 2D: X (horizontal), Z (profundidad)

        self.frame_counter = 0
        self.min_frame_gap = 10  # Aumentado para reducir keyframes

    def extract_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        points = np.column_stack(np.where(edges > 0))  # (y, x)
        points = points[:, [1, 0]]  # Convertir a (x, z)
        if len(points) > 300:
            idx = np.random.choice(len(points), 300, replace=False)
            points = points[idx]
        return points

    def icp(self, source, target, max_iterations=10):
        src = source.copy()
        tgt = target.copy()
        transform = np.eye(3)

        for _ in range(max_iterations):
            neigh = NearestNeighbors(n_neighbors=1).fit(tgt)
            distances, indices = neigh.kneighbors(src)
            matched = tgt[indices[:, 0]]

            src_mean = np.mean(src, axis=0)
            matched_mean = np.mean(matched, axis=0)
            src_centered = src - src_mean
            matched_centered = matched - matched_mean

            H = src_centered.T @ matched_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            t = matched_mean - R @ src_mean

            T = np.eye(3)
            T[:2, :2] = R
            T[:2, 2] = t

            src = (R @ src.T).T + t
            transform = T @ transform

        return transform

    def process_frame(self, frame):
        current_edges = self.extract_edges(frame)

        if self.previous_edges is None:
            self.previous_edges = current_edges
            self.keyframe_poses.append(np.eye(3))
            return

        if len(current_edges) >= 50 and len(self.previous_edges) >= 50:
            relative_transform = self.icp(current_edges, self.previous_edges)
            current_pose = self.previous_pose @ relative_transform

            if self.frame_counter >= self.min_frame_gap:
                self.keyframe_poses.append(current_pose.copy())
                self.relative_transformations.append(relative_transform.copy())
                self.previous_edges = current_edges
                self.previous_pose = current_pose
                self.frame_counter = 0
            else:
                self.frame_counter += 1

    def optimize_pose_graph(self):
        poses = [pose[:2, 2] for pose in self.keyframe_poses]  # Extraer X y Z
        keyframe_positions = np.array(poses)
        initial_parameters = keyframe_positions.flatten()

        def residual_function(params):
            residuals = []
            positions = params.reshape((-1, 2))
            for i in range(1, len(positions)):
                predicted = positions[i] - positions[i - 1]
                measured = self.relative_transformations[i - 1][:2, 2]
                residuals.extend((predicted - measured).tolist())
            return residuals

        result = least_squares(residual_function, initial_parameters, verbose=0)
        optimized = result.x.reshape((-1, 2))
        return optimized

    def save_trajectory_outputs(self, trajectory, input_video_path):
        tipo_lms = self.name
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        output_dir = os.path.join("resultados", tipo_lms, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, f"trayectoria_{self.name}")

        with open(output_base + ".csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Z"])
            writer.writerows(trajectory)

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
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

        trajectory = self.optimize_pose_graph()
        self.save_trajectory_outputs(trajectory, video_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 main.py <ruta_al_video>")
        sys.exit(1)

    input_video_path = sys.argv[1]

    if not os.path.exists(input_video_path):
        print(f"Error: no se encontró el archivo {input_video_path}")
        sys.exit(1)

    slam_system = ICPVisualSLAM()
    slam_system.process_video_input(input_video_path)
    print("Optimización global completada y trayectoria guardada.")
