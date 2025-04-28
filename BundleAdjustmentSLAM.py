import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def extract_features(img):
    orb = cv2.ORB_create(2000)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def match_features(des1, des2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def compute_pose(kp1, kp2, matches, K):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    # Matriz de calibración ficticia (modifícala si conoces los parámetros reales)
    K = np.array([[700, 0, 320],
                  [0, 700, 240],
                  [0,   0,   1]])

    poses = [np.eye(4)]

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = extract_features(prev_gray)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 500:  # limitar cuadros para velocidad
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = extract_features(gray)
        matches = match_features(des1, des2)

        if len(matches) > 30:
            R, t = compute_pose(kp1, kp2, matches, K)

            last_pose = poses[-1]
            new_pose = np.eye(4)
            new_pose[:3, :3] = R
            new_pose[:3, 3] = t.ravel()

            global_pose = last_pose @ new_pose
            poses.append(global_pose)

            kp1, des1 = kp2, des2

        frame_count += 1

    cap.release()

    # Extrae la trayectoria 2D (X, Z)
    trajectory = np.array([pose[:3, 3] for pose in poses])
    x = trajectory[:, 0]
    z = trajectory[:, 2]

    # Muestra el mapa 2D
    plt.figure(figsize=(10, 6))
    plt.plot(x, z, marker='o', linewidth=2, color='blue', label="Trayectoria")
    plt.title("Mapa 2D de trayectoria (vista superior)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main("/home/jimena/Escritorio/PROYECTO/monocular/aparta.mp4")
