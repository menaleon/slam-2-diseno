import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# === CONFIGURACIÓN ===
video_path = '/home/jimena/Escritorio/PROYECTO/monocular/aparta2.mp4'
trajectory = []
keyframe_interval = 4  # submuestreo de frames

# === MATRIZ INTRÍNSECA ESTIMADA ===
K = np.array([[700.0, 0, 360.0],
              [0, 700.0, 640.0],
              [0, 0, 1]], dtype=np.float32)

# === INICIALIZACIÓN ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video en: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

ret, first_frame = cap.read()
if not ret:
    raise ValueError("No se pudo leer el primer frame del video.")

gray_ref = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
pose = np.eye(4)
trajectory.append(pose[:3, 3])
frame_idx = 1

# === PROCESAMIENTO CON KEYFRAMES ===
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

    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good_matches) >= 8:
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts_cur = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        E, _ = cv2.findEssentialMat(pts_cur, pts_ref, K, method=cv2.RANSAC, threshold=1.0)
        if E is None:
            frame_idx += 1
            continue

        _, R, t, _ = cv2.recoverPose(E, pts_cur, pts_ref, K)
        t[1] = 0  # Eliminar movimiento vertical

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        pose = pose @ np.linalg.inv(T)

        # Solo actualizamos keyframe cada cierto intervalo
        if frame_idx % keyframe_interval == 0:
            trajectory.append(pose[:3, 3])
            kp_ref, des_ref = kp, des

    frame_idx += 1

cap.release()

# === CONVERTIR Y SUAVIZAR TRAYECTORIA ===
trajectory = np.array(trajectory)
x = gaussian_filter1d(trajectory[:, 0], sigma=2)
z = gaussian_filter1d(trajectory[:, 2], sigma=2)

# === VISUALIZACIÓN 2D PLANA (X-Z) CON MARCAS TEMPORALES ===
plt.figure(figsize=(8, 6))
plt.plot(x, z, linestyle='-', marker='o', color='blue', label='Trayectoria')

# Punto de inicio
plt.plot(x[0], z[0], marker='o', color='green', markersize=10, label='Inicio')
plt.text(x[0], z[0], 'Inicio', fontsize=9, verticalalignment='bottom', horizontalalignment='right', color='green')

# Punto de fin
plt.plot(x[-1], z[-1], marker='o', color='red', markersize=10, label='Fin')
plt.text(x[-1], z[-1], 'Fin', fontsize=9, verticalalignment='bottom', horizontalalignment='left', color='red')

# === MARCAS TEMPORALES ===
n_marcas = 6
indices_marca = np.linspace(1, len(x) - 2, n_marcas, dtype=int)

for idx in indices_marca:
    frame_real = idx * keyframe_interval
    tiempo_seg = frame_real / fps
    minutos = int(tiempo_seg // 60)
    segundos = int(tiempo_seg % 60)
    tiempo_str = f'{minutos}:{segundos:02d}'

    plt.plot(x[idx], z[idx], marker='x', color='black')
    plt.text(x[idx], z[idx], tiempo_str, fontsize=8, color='black', verticalalignment='top')

plt.title('Trayectoria estimada mejorada (Plano X-Z)')
plt.xlabel('X')
plt.ylabel('Z')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
