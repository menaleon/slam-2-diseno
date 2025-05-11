import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
video_path = '/home/jimena/Escritorio/PROYECTO/monocular/apartaJus.mp4'
voc_poses = []  # Lista para guardar la trayectoria en 2D

# === INICIALIZACIÓN ===
cap = cv2.VideoCapture(video_path)
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, first_frame = cap.read()
if not ret:
    raise ValueError("No se pudo leer el primer frame del video.")

gray_ref = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
pose = np.eye(4)
voc_poses.append(pose[:2, 3])  # Primera posición (0, 0)

K = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=np.float32)  # Matriz intrínseca dummy

# === PROCESAMIENTO FRAME A FRAME ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 10:
        continue

    matches = bf.match(des_ref, des)
    matches = sorted(matches, key=lambda x: x.distance)[:100]

    if len(matches) >= 8:
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts_cur = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        E, _ = cv2.findEssentialMat(pts_cur, pts_ref, K, method=cv2.RANSAC, threshold=1.0)
        if E is None:
            continue

        _, R, t, _ = cv2.recoverPose(E, pts_cur, pts_ref, K)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        pose = pose @ np.linalg.inv(T)
        voc_poses.append(pose[:2, 3])

        kp_ref, des_ref = kp, des

cap.release()

# === VISUALIZACIÓN TRAZADO 2D ===
voc_poses = np.array(voc_poses)
plt.figure(figsize=(6, 6))
plt.plot(voc_poses[:, 0], voc_poses[:, 1], marker='o')
plt.title("Trayectoria estimada (proyección XY)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid(True)
plt.show()
