import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import csv

class PoseGraphSLAM:
    def __init__(self, name="pose_graph_slam"):
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = np.array([[700, 0, 320],
                           [0, 700, 240],
                           [0,   0,   1]])
        self.name = name
        self.keyframes = []
        self.rel_poses = []
        self.prev_kf_img = None
        self.prev_kf_kp = None
        self.prev_kf_des = None
        self.prev_kf_pose = np.eye(4)
        self.frame_count = 0
        self.min_gap = 5
        self.min_translation = 0.05

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

        if self.prev_kf_des is not None and des is not None and len(kp) > 0:
            matches = self.lowe_ratio_match(self.prev_kf_des, des)
            if len(matches) > 30:
                pts1 = np.float32([self.prev_kf_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
                    rel_pose = np.eye(4)
                    rel_pose[:3, :3] = R
                    rel_pose[:3, 3] = t.ravel()
                    new_pose = self.prev_kf_pose @ rel_pose

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
            self.keyframes.append(np.eye(4))
            self.prev_kf_kp = kp
            self.prev_kf_des = des
            self.prev_kf_pose = np.eye(4)

    def pose_graph_optimization(self):
        poses = np.array([kf[:3, 3] for kf in self.keyframes])  # Solo posiciones (X,Y,Z)
        x0 = poses.flatten()

        def residuals(x):
            res = []
            poses_opt = x.reshape((-1, 3))
            for i in range(1, len(poses_opt)):
                pred = poses_opt[i] - poses_opt[i-1]
                rel = self.rel_poses[i-1][:3, 3]
                res.extend((pred - rel).tolist())
            return res

        result = least_squares(residuals, x0, verbose=0)
        optimized_positions = result.x.reshape((-1, 3))
        return optimized_positions[:, [0, 2]]  # Solo X-Z

    def save_outputs(self, trajectory):
        base = f"trayectoria_{self.name}"
        with open(base + ".csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Z"])
            writer.writerows(trajectory)

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
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

        optimized_trajectory = self.pose_graph_optimization()
        self.save_outputs(optimized_trajectory)


# ---------- Ejecución ----------
if __name__ == "__main__":
    video_path = "/home/jimena/Escritorio/PROYECTO/monocular/aparta.mp4"
    slam = PoseGraphSLAM()
    slam.process_video(video_path)
    print("Optimización global completada y trayectoria guardada.")
