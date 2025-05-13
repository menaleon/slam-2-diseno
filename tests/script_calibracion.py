import cv2
import numpy as np
import glob
import yaml
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DEL TABLERO DE AJEDREZ ===
pattern_size = (6, 9)  # 7 columnas × 10 filas de cuadros o sea 6 × 9 esquinas internas
square_size = 0.025  # tamaño estimado del cuadro en metros

# === PREPARAR PUNTOS OBJETO 3D ===
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # puntos 3D reales
imgpoints = []  # puntos 2D detectados

# === CARGAR IMÁGENES (.jpg y .JPG) ===
images = glob.glob('calibration_images/*.jpg') + glob.glob('calibration_images/*.JPG')
print(f"Total de imágenes encontradas: {len(images)}")

if not images:
    raise FileNotFoundError("No se encontraron imágenes en calibration_images/*.jpg o *.JPG")

# === PROCESAR IMÁGENES CON MEJORAS DE DETECCIÓN ===
valid_images = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mejorar contraste
    gray = cv2.equalizeHist(gray)

    # Detección robusta
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        valid_images += 1

        img_vis = cv2.drawChessboardCorners(img.copy(), pattern_size, corners, ret)
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Patrón detectado: {fname}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Patrón NO detectado en: {fname}")

print(f"\nTotal de imágenes válidas: {valid_images}")

# === CALIBRACIÓN DE LA CÁMARA ===
if valid_images == 0:
    raise RuntimeError("No se detectó el patrón en ninguna imagen. Asegúrate de usar pattern_size = (6, 9).")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\nMATRIZ INTRÍNSECA (K):")
print(K)
print("\nCOEFICIENTES DE DISTORSIÓN:")
print(dist.ravel())

# === GUARDAR RESULTADO ===
calib_data = {
    'camera_matrix': K.tolist(),
    'dist_coeff': dist.tolist()
}

with open("camera_calibration.yaml", "w") as f:
    yaml.dump(calib_data, f)

print("\nParámetros guardados en camera_calibration.yaml")
