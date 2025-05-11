import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
video_path = '/home/jimena/Escritorio/PROYECTO/monocular/aparta.mp4'
trajectory = []
orientations = []

# === MATRIZ INTRÍNSECA ESTIMADA ===
"""
K = np.array([[630.0, 0, 360.0],
              [0, 630.0, 640.0],
              [0, 0, 1]], dtype=np.float32)
              """

K = np.array([
    [914.3,   0.0, 640.0],
    [  0.0, 929.0, 360.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

# === INICIALIZACIÓN ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video en: {video_path}")

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, first_frame = cap.read()
if not ret:
    raise ValueError("No se pudo leer el primer frame del video.")

gray_ref = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
pose = np.eye(4)
trajectory.append(pose[:3, 3])
orientations.append(pose[:3, 2])  # El eje Z de la cámara indica hacia dónde "mira"

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
        trajectory.append(pose[:3, 3])
        orientations.append(pose[:3, 2])  # eje Z en coordenadas de mundo

        kp_ref, des_ref = kp, des

cap.release()

# === VISUALIZACIÓN CON FLECHAS ===
trajectory = np.array(trajectory)
orientations = np.array(orientations)

x = trajectory[:, 0]
z = trajectory[:, 2]
u = orientations[:, 0]
w = orientations[:, 2]
                                                
# Visualización con inicio y fin marcados
plt.figure(figsize=(8, 6))
plt.quiver(x, z, u, w, angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)
plt.plot(x, z, marker='o', linestyle='--', color='gray', alpha=0.5)

# Marcar punto de inicio
plt.plot(x[0], z[0], marker='o', color='green', markersize=10, label='Inicio')
plt.text(x[0], z[0], 'Inicio', fontsize=9, verticalalignment='bottom', horizontalalignment='right', color='green')

# Marcar punto final
plt.plot(x[-1], z[-1], marker='o', color='red', markersize=10, label='Fin')
plt.text(x[-1], z[-1], 'Fin', fontsize=9, verticalalignment='bottom', horizontalalignment='left', color='red')

plt.title("Trayectoria estimada con orientación (Plano X-Z)")
plt.xlabel("X")
plt.ylabel("Z")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
