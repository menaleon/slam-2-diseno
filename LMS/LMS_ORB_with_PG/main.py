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
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.name = Path(__file__).resolve().parent.name

        self.orb_detector = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        self.keyframe_poses = []
        self.relative_transformations = []

        self.previous_keyframe_image = None
        self.previous_keyframe_keypoints = None
        self.previous_keyframe_descriptors = None
        self.previous_keyframe_pose = np.eye(4)

        self.frame_counter = 0
        self.min_frame_gap = 5
        self.min_keyframe_translation = 0.05

        # Métricas para evaluación
        self.total_successful_frames = 0
        self.total_tracked_matches = 0
        self.total_translation_magnitude = 0.0
        self.total_pose_estimations = 0

    def filter_matches_lowe_ratio(self, descriptors1, descriptors2, ratio=0.75):
        knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches

    def process_frame(self, frame):
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb_detector.detectAndCompute(grayscale_frame, None)

        if self.previous_keyframe_descriptors is not None and descriptors is not None and len(keypoints) > 0:
            matches = self.filter_matches_lowe_ratio(self.previous_keyframe_descriptors, descriptors)

            if len(matches) > 30:
                points_prev = np.float32([self.previous_keyframe_keypoints[m.queryIdx].pt for m in matches])
                points_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

                essential_matrix, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix, method=cv2.RANSAC, threshold=1.0)

                if essential_matrix is not None:
                    _, rotation, translation, _ = cv2.recoverPose(essential_matrix, points_prev, points_curr, self.camera_matrix)

                    relative_pose = np.eye(4)
                    relative_pose[:3, :3] = rotation
                    relative_pose[:3, 3] = translation.ravel()

                    current_pose = self.previous_keyframe_pose @ relative_pose
                    translation_magnitude = np.linalg.norm(relative_pose[:3, 3])

                    if self.frame_counter >= self.min_frame_gap or translation_magnitude > self.min_keyframe_translation:
                        self.keyframe_poses.append(current_pose.copy())
                        self.relative_transformations.append(relative_pose.copy())
                        self.previous_keyframe_keypoints = keypoints
                        self.previous_keyframe_descriptors = descriptors
                        self.previous_keyframe_pose = current_pose
                        self.frame_counter = 0
                    else:
                        self.frame_counter += 1

                    # Actualizar métricas
                    self.total_successful_frames += 1
                    self.total_tracked_matches += len(matches)
                    self.total_translation_magnitude += translation_magnitude
                    self.total_pose_estimations += 1
        else:
            self.keyframe_poses.append(np.eye(4))
            self.previous_keyframe_keypoints = keypoints
            self.previous_keyframe_descriptors = descriptors
            self.previous_keyframe_pose = np.eye(4)

    def optimize_pose_graph(self):
        keyframe_positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])
        initial_parameters = keyframe_positions.flatten()

        def residual_function(parameters):
            residuals = []
            optimized_positions = parameters.reshape((-1, 3))
            for i in range(1, len(optimized_positions)):
                predicted_motion = optimized_positions[i] - optimized_positions[i - 1]
                measured_motion = self.relative_transformations[i - 1][:3, 3]
                residuals.extend((predicted_motion - measured_motion).tolist())
            return residuals

        result = least_squares(residual_function, initial_parameters, verbose=0)
        optimized_positions = result.x.reshape((-1, 3))
        return optimized_positions[:, [0, 2]]

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

        # Cálculo de métricas
        num_keyframes = len(self.keyframe_poses)
        avg_translation = self.total_translation_magnitude / max(1, self.total_pose_estimations)
        avg_matches = self.total_tracked_matches / max(1, self.total_pose_estimations)
        triangulation_success_rate = self.total_successful_frames / max(1, self.total_pose_estimations)

        # Graficar trayectoria
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Optimized Trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color='g', label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='r', label='End')

        # Leyenda de métricas
        info_text = (
            f"Keyframes: {num_keyframes}\n"
            f"Prom. matches por pose: {avg_matches:.1f}\n"
            f"Éxito triangulación: {triangulation_success_rate:.2%}\n"
            f"Mov. medio entre keyframes: {avg_translation:.2f} m"
        )
        plt.gcf().text(0.02, 0.72, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.title(f"{self.name} para {Path(input_video_path).name}")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_base + ".png")
        plt.close()

    def process_video_input(self, video_path):
        video_capture = cv2.VideoCapture(video_path)

        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
            self.process_frame(frame)

        video_capture.release()

        optimized_trajectory = self.optimize_pose_graph()
        self.save_trajectory_outputs(optimized_trajectory, video_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 main.py <ruta_al_video>")
        sys.exit(1)

    input_video_path = sys.argv[1]

    if not os.path.exists(input_video_path):
        print(f"Error: no se encontró el archivo {input_video_path}")
        sys.exit(1)

    slam_system = PoseGraphSLAM()
    slam_system.process_video_input(input_video_path)
    print("Optimizacion global completada y trayectoria guardada.")
