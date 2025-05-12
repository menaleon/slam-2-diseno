import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# === CONFIGURACIÓN ===
video_path = '/home/jimena/Escritorio/PROYECTO/monocular/aparta.mp4'
keyframe_interval = 10

K = np.array([[700.0, 0, 360.0],
              [0, 700.0, 640.0],
              [0, 0, 1]], dtype=np.float32)

# === INICIALIZACIÓN ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video en: {video_path}")

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

ret, first_frame = cap.read()
if not ret:
    raise ValueError("No se pudo leer el primer frame del video.")

gray_ref = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
pose = np.eye(4)

trajectory = [pose[:3, 3]]
keyframes = [{
    'pose': pose.copy(),
    'keypoints': kp_ref,
    'descriptors': des_ref
}]
frame_idx = 1

# === LOOP FRAME A FRAME ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 10:
        frame_idx += 1
        continue

    matches = bf.knnMatch(des_ref, des, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) >= 8:
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts_cur = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        E, _ = cv2.findEssentialMat(pts_cur, pts_ref, K, method=cv2.RANSAC, threshold=1.0)
        if E is None:
            frame_idx += 1
            continue

        _, R, t, _ = cv2.recoverPose(E, pts_cur, pts_ref, K)
        t[1] = 0  # restringe el movimiento al plano X-Z

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        pose = pose @ np.linalg.inv(T)

        if frame_idx % keyframe_interval == 0:
            trajectory.append(pose[:3, 3])
            keyframes.append({
                'pose': pose.copy(),
                'keypoints': kp,
                'descriptors': des
            })
            kp_ref, des_ref = kp, des

    frame_idx += 1

cap.release()

# === TRIANGULACIÓN ENTRE DOS PRIMEROS KEYFRAMES CON MATCHES VALIDOS ===
def triangulate_points(K, kf1, kf2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(kf1['descriptors'], kf2['descriptors'], k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    pts1 = np.float32([kf1['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([kf2['keypoints'][m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    proj1 = K @ kf1['pose'][:3]
    proj2 = K @ kf2['pose'][:3]

    pts4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    return pts3d, pts1, pts2

pts3d, pts1, pts2 = triangulate_points(K, keyframes[0], keyframes[1])

# === OBSERVACIONES 2D PARA BA ===
observations = []
for i, pt2d in enumerate(pts1):
    observations.append((0, i, pt2d))
for i, pt2d in enumerate(pts2):
    observations.append((1, i, pt2d))

# === CONVERTIR POSES A FORMATO rvec + tvec ===
def pose_to_rt(pose):
    R = pose[:3, :3]
    t = pose[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return np.hstack([rvec.ravel(), t.ravel()])

poses_init = np.array([pose_to_rt(kf['pose']) for kf in keyframes[:2]])
params_init = np.hstack([poses_init.ravel(), pts3d.ravel()])

# === FUNCION DE ERROR PARA BA ===
def reprojection_residuals(params, n_poses, n_points, K, observations):
    poses = params[:n_poses * 6].reshape((n_poses, 6))
    points_3d = params[n_poses * 6:].reshape((n_points, 3))
    residuals = []

    for pose_idx, pt_idx, pt2d in observations:
        rvec = poses[pose_idx, :3]
        tvec = poses[pose_idx, 3:].reshape((3, 1))
        pt3d = points_3d[pt_idx].reshape((1, 3))

        pt2d_proj, _ = cv2.projectPoints(pt3d, rvec, tvec, K, distCoeffs=None)
        residuals.append((pt2d_proj.ravel() - pt2d).ravel())

    return np.concatenate(residuals)

# === OPTIMIZACIÓN CON BUNDLE ADJUSTMENT ===
n_poses = 2
n_points = pts3d.shape[0]

res = least_squares(
    reprojection_residuals,
    params_init,
    verbose=2,
    x_scale='jac',
    ftol=1e-4,
    method='lm',
    args=(n_poses, n_points, K, observations)
)

# === EXTRAER TRAJECTORIA AJUSTADA ===
params_optimized = res.x
poses_opt = params_optimized[:n_poses * 6].reshape((n_poses, 6))

trajectory_ba = []
for pose_rt in poses_opt:
    rvec = pose_rt[:3]
    tvec = pose_rt[3:].reshape((3, 1))
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    trajectory_ba.append(T[:3, 3])

trajectory_ba = np.array(trajectory_ba)
trajectory = np.array(trajectory)

# === VISUALIZACIÓN COMPARATIVA ===
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 2], 'o--', label='Sin BA', alpha=0.6)
plt.plot(trajectory_ba[:, 0], trajectory_ba[:, 2], 'o-', label='Con BA', color='blue')
plt.plot(trajectory_ba[0, 0], trajectory_ba[0, 2], 'go', label='Inicio')
plt.plot(trajectory_ba[-1, 0], trajectory_ba[-1, 2], 'ro', label='Fin')

plt.title("Trayectoria estimada (X-Z) con y sin Bundle Adjustment")
plt.xlabel("X")
plt.ylabel("Z")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
