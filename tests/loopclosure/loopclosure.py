import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from collections import deque
import csv

class LoopClosureSLAM:
    def __init__(self, name="loop_closure_slam"):
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = np.array([[700, 0, 320],
                           [0, 700, 240],
                           [0,   0,   1]])
        self.name = name

        self.keyframes = []        # lista de poses 4x4
        self.images = []           # lista de imágenes grises
        self.kps = []              # lista de keypoints
        self.descs = []            # lista de descriptores
        self.rel_poses = []        # transformaciones entre keyframes
        self.constraints = []      # pares (i, j, relative_transform)
        self.last_pose = np.eye(4)

        self.min_gap = 5
        self.min_translation = 0.05
        self.frame_count = 0

    def lowe_ratio_match(self, des1, des2, ratio=0.75):
        knn_matches = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    def detect_loop_closure(self, curr_des):
        threshold_matches = 40
        min_index_gap = 10

        for i, prev_des in enumerate(self.descs[:-min_index_gap]):
            matches = self.lowe_ratio_match(prev_des, curr_des)
            if len(matches) > threshold_matches:
                return i  # loop closure encontrado con el keyframe i
        return None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if len(self.keyframes) == 0:
            self.keyframes.append(np.eye(4))
            self.images.append(gray)
            self.kps.append(kp)
            self.descs.append(des)
            self.last_pose = np.eye(4)
            return

        if des is not None and self.descs[-1] is not None:
            matches = self.lowe_ratio_match(self.descs[-1], des)
            if len(matches) > 30:
                pts1 = np.float32([self.kps[-1][m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
                    rel_pose = np.eye(4)
                    rel_pose[:3, :3] = R
                    rel_pose[:3, 3] = t.ravel()

                    global_pose = self.last_pose @ rel_pose
                    delta = np.linalg.norm(rel_pose[:3, 3])

                    if self.frame_count >= self.min_gap or delta > self.min_translation:
                        # Agregar keyframe normal
                        self.keyframes.append(global_pose)
                        self.images.append(gray)
                        self.kps.append(kp)
                        self.descs.append(des)
                        self.rel_poses.append(rel_pose)
                        self.constraints.append((len(self.keyframes)-2, len(self.keyframes)-1, rel_pose))
                        self.last_pose = global_pose
                        self.frame_count = 0

                        # Intentar detección de loop closure
                        lc_index = self.detect_loop_closure(des)
                        if lc_index is not None:
                            #print(f"[LOOP CLOSURE] Detected with keyframe {lc_index}")
                            # Añadir nueva restricción entre i y j
                            pose_i = self.keyframes[lc_index]
                            rel_lc = np.linalg.inv(pose_i) @ global_pose
                            self.constraints.append((lc_index, len(self.keyframes)-1, rel_lc))
                    else:
                        self.frame_count += 1

    def optimize_pose_graph(self):
        poses = np.array([kf[:3, 3] for kf in self.keyframes])  # Solo posiciones
        x0 = poses.flatten()

        def residuals(x):
            poses_opt = x.reshape((-1, 3))
            res = []
            for i, j, rel in self.constraints:
                pred = poses_opt[j] - poses_opt[i]
                rel_t = rel[:3, 3]
                res.extend((pred - rel_t).tolist())
            return res

        result = least_squares(residuals, x0, verbose=0)
        optimized = result.x.reshape((-1, 3))
        return optimized[:, [0, 2]]  # X-Z

    def save_outputs(self, trajectory):
        base = f"trayectoria_{self.name}"
        with open(base + ".csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Z"])
            writer.writerows(trajectory)

        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Optimized with Loop Closure')
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

        optimized_traj = self.optimize_pose_graph()
        self.save_outputs(optimized_traj)


# ---------- Ejecución ----------
if __name__ == "__main__":
    video_path = "/home/jimena/Escritorio/PROYECTO/monocular/apartaJus.mp4"
    slam = LoopClosureSLAM()
    slam.process_video(video_path)
    print("SLAM con cierre de lazo completado.")
