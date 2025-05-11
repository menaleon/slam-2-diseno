import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN INICIAL ===
video_path = '/home/jimena/Escritorio/PROYECTO/monocular/apartaJus.mp4'
min_matches = 100  # Numero mínimo de matches buenos requeridos para estimar la pose

# === FUNCIONES DE APOYO ===
def draw_matches(img1, kp1, img2, kp2, matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

def extract_orb_features(img, orb):
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

# === ETAPA DE INICIALIZACIÓN (FRAME 0 y FRAME 1) ===
cap = cv2.VideoCapture(video_path)
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret1, frame1 = cap.read()
ret2, frame2 = cap.read()

if not ret1 or not ret2:
    raise ValueError("No se pudieron leer los primeros dos frames del video.")

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

kp1, des1 = extract_orb_features(gray1, orb)
kp2, des2 = extract_orb_features(gray2, orb)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Filtrar puntos buenos
if len(matches) > min_matches:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)

    print("Matriz de rotación R:")
    print(R)
    print("\nVector de traslación t:")
    print(t)

    # Visualización
    match_img = draw_matches(gray1, kp1, gray2, kp2, matches)
    plt.imshow(match_img)
    plt.title("Matches ORB entre Frame 0 y Frame 1")
    plt.axis("off")
    plt.show()
else:
    print("No se encontraron suficientes correspondencias buenas.")

cap.release()
