import cv2
import numpy as np
import open3d as o3d

# Cargar el video
video_path = "/home/jimena/Descargas/car.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

print("Video cargado correctamente.")

# ORB Feature Detector
orb = cv2.ORB_create(500)
print("Detector ORB inicializado.")

# Parámetros de la cámara (suponiendo cámara genérica)
K = np.array([[700, 0, 320],
              [0, 700, 240],
              [0, 0, 1]])

# Lista de nubes de puntos
point_cloud = []

# Leer el primer fotograma
ret, prev_frame = cap.read()
if not ret:
    print("Error: No se pudo leer el primer fotograma.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(prev_gray, None)

if kp1 is None or des1 is None:
    print("No se encontraron características en el primer fotograma.")
    exit()

print(f"Se detectaron {len(kp1)} características en el primer fotograma.")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del video alcanzado.")
        break

    frame_count += 1
    print(f"\n Procesando fotograma {frame_count}...")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)

    if kp2 is None or des2 is None:
        print("No se encontraron características en este fotograma.")
        continue

    print(f"Características detectadas en el fotograma {frame_count}: {len(kp2)}")

    # Matcher de características
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        print("Pocas coincidencias encontradas, saltando este fotograma.")
        continue

    matches = sorted(matches, key=lambda x: x.distance)
    print(f"{len(matches)} coincidencias encontradas entre fotogramas.")

    # Extraer puntos coincidentes
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calcular la matriz esencial
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None:
        print("⚠️ No se pudo calcular la matriz esencial. Saltando este fotograma.")
        continue

    print("Matriz esencial calculada.")

    # Recuperar la pose de la cámara
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    print(f"Pose estimada: Rotación:\n{R}\nTraslación:\n{t.T}")

    # Triangulación de puntos 3D
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Primera pose en (0,0,0)
    P2 = np.hstack((R, t))  # Segunda pose estimada

    points_4d_hom = cv2.triangulatePoints(K @ P1, K @ P2, pts1, pts2)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]

    print(f"Se triangularon {points_3d.shape[1]} puntos 3D.")

    # Agregar puntos a la nube
    point_cloud.append(points_3d.T)

    # Actualizar fotograma anterior
    kp1, des1 = kp2, des2

cap.release()

# Convertir puntos a formato Open3D
if len(point_cloud) > 0:
    print("\nProcesando la nube de puntos...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(point_cloud))
    
    print(f"Nube de puntos generada con {len(pcd.points)} puntos.")
    
    # Visualizar mapa 3D
    print("Mostrando el mapa 3D...")
    o3d.visualization.draw_geometries([pcd])
else:
    print("No se generó suficiente información para construir la nube de puntos.")
