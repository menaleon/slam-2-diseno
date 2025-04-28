import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import csv

class LoopClosureVerifiedSLAM:
    def __init__(self, name="loop_closure_verified_slam"):
        self.name = name
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = np.array([[700, 0, 320],
                           [0, 700, 240],
                           [0,   0,   1]])
        self.keyframes = []
        self.images = []
        self.kps = []
        self.descs = []
        self.rel_poses = []
        self.constraints = []
        self.last_pose = np.eye(4)
        self.min_gap = 5
        self.min_translation = 0.05
        self.min_index_gap = 10
        self.frame_count = 0

    def lowe_ratio_match(self, des1, des2, ratio=0.75):
        knn_matches = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    def verify_geometric_consistency(self, kp1, kp2, matches):
        if len(matches) < 30:
            return False, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, threshold=1.0)
        if E is None or mask.sum() < 20:
            return False, None
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t.ravel()
        return True, rel_pose

    def detect_loop_closure(self, curr_kp, curr_des):
        threshold_matches = 40
        for i, (prev_kp, prev_des) in enumerate(zip(self.kps[:-self.min_index_gap], self.descs[:-self.min_index_gap])):
            matches = self.lowe_ratio_match(prev_des, curr_des)
            if len(matches) > threshold_matches:
                consistent, rel_pose = self.verify_geometric_consistency(prev_kp, curr_kp, matches)
                if consistent:
                    return i, rel_pose
        return None, None

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
                        self.keyframes.append(global_pose)
                        self.images.append(gray)
                        self.kps.append(kp)
                        self.descs.append(des)
                        self.rel_poses.append(rel_pose)
                        self.constraints.append((len(self.keyframes)-2, len(self.keyframes)-1, rel_pose))
                        self.last_pose = global_pose
                        self.frame_count = 0

                        # Detección de loop closure con verificación
                        loop_idx, rel_lc = self.detect_loop_closure(kp, des)
                        if loop_idx is not None:
                            #print(f"[LOOP CLOSURE VERIFICADO] con keyframe {loop_idx}")
                            self.constraints.append((loop_idx, len(self.keyframes)-1, rel_lc))
                    else:
                        self.frame_count += 1

    def optimize_pose_graph(self):
        poses = np.array([kf[:3, 3] for kf in self.keyframes])
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
        return optimized[:, [0, 2]]

    def save_outputs(self, trajectory):
        base = f"trayectoria_{self.name}"
        with open(base + ".csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Z"])
            writer.writerows(trajectory)

        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Loop Closure Verificado')
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
        trajectory = self.optimize_pose_graph()
        self.save_outputs(trajectory)


# ---------- Ejecución ----------
if __name__ == "__main__":
    video_path = "/home/jimena/Escritorio/PROYECTO/monocular/aparta.mp4"
    slam = LoopClosureVerifiedSLAM()
    slam.process_video(video_path)
    print("SLAM con cierre de lazo verificado completado.")
