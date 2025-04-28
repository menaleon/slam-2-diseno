import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter
import csv
import os

class KeyframeVisualSLAM3D:
    def __init__(self, name="KeyframeVisualSLAM"):
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = np.array([[700, 0, 320],
                           [0, 700, 240],
                           [0,   0,   1]])
        self.name = name
        self.last_kf_kp = None
        self.last_kf_des = None
        self.last_kf_pose = np.eye(4)
        self.trajectory = deque()
        self.poses = [np.eye(4)]
        self.frame_count = 0
        self.min_frame_gap = 5
        self.min_translation = 0.05  # mínimo movimiento para considerar nuevo keyframe

    def add_keyframe(self, pose):
        self.poses.append(pose)
        self.trajectory.append(pose[:3, 3][[0, 2]])  # X-Z

    def is_keyframe(self, pose):
        delta = np.linalg.norm(pose[:3, 3] - self.last_kf_pose[:3, 3])
        return self.frame_count >= self.min_frame_gap or delta > self.min_translation

    def lowe_ratio_match(self, des1, des2, ratio=0.75):
        knn_matches = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.last_kf_des is not None and des is not None and len(kp) > 0:
            matches = self.lowe_ratio_match(self.last_kf_des, des)
            if len(matches) > 30:
                pts1 = np.float32([self.last_kf_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
                    new_pose = np.eye(4)
                    new_pose[:3, :3] = R
                    new_pose[:3, 3] = t.ravel()

                    # Pose acumulada desde el último keyframe
                    global_pose = self.last_kf_pose @ new_pose

                    if self.is_keyframe(global_pose):
                        self.add_keyframe(global_pose)
                        self.last_kf_kp = kp
                        self.last_kf_des = des
                        self.last_kf_pose = global_pose
                        self.frame_count = 0
                    else:
                        self.frame_count += 1
        else:
            # Primer frame como keyframe
            self.last_kf_kp = kp
            self.last_kf_des = des
            self.last_kf_pose = np.eye(4)
            self.add_keyframe(np.eye(4))

    def smooth_trajectory(self):
        if len(self.trajectory) > 5:
            traj_array = np.array(self.trajectory)
            window_size = min(11, len(traj_array) - 1 if len(traj_array) % 2 == 0 else len(traj_array))
            smoothed_x = savgol_filter(traj_array[:, 0], window_size, 3)
            smoothed_z = savgol_filter(traj_array[:, 1], window_size, 3)
            return np.vstack((smoothed_x, smoothed_z)).T
        return np.array(self.trajectory)

    def save_outputs(self):
        traj = self.smooth_trajectory()
        base = f"trayectoria_{self.name}"

        # CSV
        with open(base + ".csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Z"])
            writer.writerows(traj)

        # PNG
        plt.plot(traj[:, 0], traj[:, 1], 'b-', label='Smoothed Trajectory')
        plt.scatter(traj[0, 0], traj[0, 1], color='g', label='Start')
        plt.scatter(traj[-1, 0], traj[-1, 1], color='r', label='End')
        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.title(f"Trajectory 2D - {self.name}")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.savefig(base + ".png")
        plt.close()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()
        self.save_outputs()


# ---------- Ejemplo de uso ----------
if __name__ == "__main__":
    video_path = "/home/jimena/Escritorio/PROYECTO/monocular/car.mp4"
    slam = KeyframeVisualSLAM3D(name="keyframe_visualslam")
    slam.process_video(video_path)
    print("Procesamiento con keyframes completado.")
