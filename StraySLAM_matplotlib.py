import os
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
import matplotlib.pyplot as plt
import time

description = """
Este script visualiza datasets tomados con la aplicación Stray Scanner.
"""

usage = """
Llamar con: python stray_visualize.py <ruta-al-directorio-del-dataset>
"""

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('path', type=str, help="Ruta al dataset a procesar.")
    parser.add_argument('--every', type=int, default=15, help="Procesar cada 15 frames")
    return parser.parse_args()

def read_data(flags):
    intrinsics = np.loadtxt(os.path.join(flags.path, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(flags.path, 'odometry.csv'), delimiter=',', skiprows=1)
    poses = []

    for line in odometry:
        position = line[2:5]
        quaternion = line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3, 3] = position
        poses.append(T_WC)
    depth_dir = os.path.join(flags.path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir)) if p.endswith('.npy') or p.endswith('.png')]
    return { 'poses': poses, 'intrinsics': intrinsics, 'depth_frames': depth_frames }

def load_depth_image(path):
    if path.endswith('.npy'):
        depth_mm = np.load(path)
    elif path.endswith('.png'):
        depth_mm = np.array(Image.open(path))
    else:
        raise ValueError(f"Formato de archivo de profundidad no soportado: {path}")
    depth_m = depth_mm.astype(np.float32) / 1000.0  # Convertir mm a metros
    return depth_m

def compute_point_cloud_for_frame(i, data):
    depth_path = data['depth_frames'][i]
    depth = load_depth_image(depth_path)
    
    intrinsics = data['intrinsics']
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    T_WC = data['poses'][i]
    height, width = depth.shape
    
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid = points_camera[:, 2] > 0
    points_camera = points_camera[valid]
    
    R_WC, t_WC = T_WC[:3, :3], T_WC[:3, 3]
    points_world = (R_WC @ points_camera.T).T + t_WC
    return points_world

def visualize_with_matplotlib(trajectory_points, point_cloud):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(trajectory_points) > 0:
        trajectory_points = np.array(trajectory_points)
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 'r-', label='Trayectoria')
    
    if len(point_cloud) > 0:
        point_cloud = np.array(point_cloud)
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=0.5, c='g', marker='o', label='Nube de Puntos')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Visualización de SLAM con Matplotlib")
    plt.legend()
    plt.show()

def main():
    flags = read_args()
    if not os.path.exists(flags.path):
        print(f"El directorio {flags.path} no existe.")
        return

    data = read_data(flags)
    trajectory_points = []
    point_cloud_points = []
    
    for i in range(0, len(data['poses']), flags.every):
        T_WC = data['poses'][i]
        trajectory_points.append(T_WC[:3, 3])
        
        points_world = compute_point_cloud_for_frame(i, data)
        if len(points_world) > 0:
            point_cloud_points.append(points_world)
        
        time.sleep(0.05)  # Simular actualización en tiempo real
    
    if point_cloud_points:
        all_points = np.vstack(point_cloud_points)
        visualize_with_matplotlib(trajectory_points, all_points)

if __name__ == "__main__":
    main()
