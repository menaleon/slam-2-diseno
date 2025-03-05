import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter

class VisualSLAM:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)  # Aumentar la cantidad de características
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.trajectory = deque()
        self.position = np.array([0.0, 0.0], dtype=np.float64)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if self.prev_frame is not None and self.prev_descriptors is not None:
            matches = self.bf.match(self.prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 10:
                src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])

                transform_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)
                if transform_matrix is not None:
                    movement = transform_matrix[:2, 2].astype(np.float64) * 0.1  # Escalar el movimiento
                    self.position += movement
                    self.trajectory.append(self.position.copy())
        
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def smooth_trajectory(self):
        if len(self.trajectory) > 5:
            trajectory_array = np.array(self.trajectory)
            window_size = min(11, len(trajectory_array) - 1)
            smoothed_x = savgol_filter(trajectory_array[:, 0], window_size, 3)
            smoothed_y = savgol_filter(trajectory_array[:, 1], window_size, 3)
            return np.vstack((smoothed_x, smoothed_y)).T
        return np.array(self.trajectory)

    def plot_trajectory(self):
        if len(self.trajectory) > 1:
            smoothed_trajectory = self.smooth_trajectory()
            plt.plot(smoothed_trajectory[:, 0], -smoothed_trajectory[:, 1], 'b-', label='Smoothed Trajectory')
            plt.scatter(smoothed_trajectory[0, 0], -smoothed_trajectory[0, 1], color='g', label='Start')
            plt.scatter(smoothed_trajectory[-1, 0], -smoothed_trajectory[-1, 1], color='r', label='End')
            plt.legend()
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.title("Estimated Smoothed Trajectory")
            plt.show()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()
        self.plot_trajectory()

if __name__ == "__main__":
    slam = VisualSLAM()
    video_path = "/home/jimena/Descargas/aparta.mp4"  # Reemplázalo con tu video de entrada
    slam.process_video(video_path)
