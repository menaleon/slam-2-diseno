import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import csv
import os

# ---------- Clase 1: Visual SLAM monocular sin Bundle Adjustment ----------
class VisualSLAM3D:
    def __init__(self, name="VisualSLAM"):
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.K = np.array([[700, 0, 320],
                           [0, 700, 240],
                           [0,   0,   1]])
        self.name = name
        self.prev_kp = None
        self.prev_des = None
        self.prev_gray = None
        self.trajectory = deque()
        self.poses = [np.eye(4)]

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_des is not None and len(kp) > 0:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 30:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
                    last_pose = self.poses[-1]
                    new_pose = np.eye(4)
                    new_pose[:3, :3] = R
                    new_pose[:3, 3] = t.ravel()
                    current_pose = last_pose @ new_pose
                    self.poses.append(current_pose)

                    position = current_pose[:3, 3]
                    self.trajectory.append(np.array([position[0], position[2]]))  # X-Z

        self.prev_kp = kp
        self.prev_des = des
        self.prev_gray = gray

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


# ---------- Clase 2: Visual SLAM monocular con Bundle Adjustment ----------
class BundleAdjustmentSLAM(VisualSLAM3D):
    def __init__(self, name="BundleAdjustmentSLAM"):
        super().__init__(name)

    def bundle_adjustment(self, pts3D, pts2D, K, R_init, t_init):
        def project(R, t, X):
            X_cam = (R @ X.T + t.reshape(3, 1))
            x = X_cam[:2] / X_cam[2]
            x = (K[:2, :2] @ x + K[:2, 2:3])
            return x.T

        def residuals(params):
            rvec, tvec = params[:3], params[3:]
            R, _ = cv2.Rodrigues(rvec)
            proj_pts = project(R, tvec, pts3D)
            return (proj_pts - pts2D).ravel()

        rvec_init, _ = cv2.Rodrigues(R_init)
        x0 = np.hstack((rvec_init.ravel(), t_init.ravel()))
        result = least_squares(residuals, x0, method='lm')
        rvec_opt, tvec_opt = result.x[:3], result.x[3:]
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        return R_opt, tvec_opt

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_des is not None and len(kp) > 0:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 30:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
                    proj1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                    proj2 = self.K @ np.hstack((R, t))
                    pts4D = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
                    pts3D = (pts4D[:3] / pts4D[3]).T

                    R_opt, t_opt = self.bundle_adjustment(pts3D, pts2, self.K, R, t)

                    last_pose = self.poses[-1]
                    new_pose = np.eye(4)
                    new_pose[:3, :3] = R_opt
                    new_pose[:3, 3] = t_opt.ravel()
                    current_pose = last_pose @ new_pose
                    self.poses.append(current_pose)

                    position = current_pose[:3, 3]
                    self.trajectory.append(np.array([position[0], position[2]]))  # X-Z

        self.prev_kp = kp
        self.prev_des = des
        self.prev_gray = gray


# ---------- Uso principal ----------
if __name__ == "__main__":
    video_path = "/home/jimena/Escritorio/PROYECTO/monocular/car.mp4"

    # Procesar con VisualSLAM sin BA
    slam1 = VisualSLAM3D(name="sin_bundle_adjustment")
    slam1.process_video(video_path)

    # Procesar con SLAM con BA
    slam2 = BundleAdjustmentSLAM(name="con_bundle_adjustment")
    slam2.process_video(video_path)

    print("Ambas trayectorias fueron procesadas y guardadas como .csv y .png.")
