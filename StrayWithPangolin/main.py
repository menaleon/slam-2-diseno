import os
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
from descriptor import PangolinViewer
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
        # timestamp, frame, x, y, z, qx, qy, qz, qw
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
    # Cargar la imagen de profundidad
    depth_path = data['depth_frames'][i]
    depth = load_depth_image(depth_path)  # depth es un array 2D de valores en metros

    # Obtener las intrínsecas de la cámara
    intrinsics = data['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Obtener la pose de la cámara
    T_WC = data['poses'][i]

    # Dimensiones de la imagen
    height, width = depth.shape

    # Crear una rejilla de coordenadas de píxeles
    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)

    # Aplanar los arrays
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    depth_flat = depth.flatten()

    # Filtrar valores de profundidad válidos
    valid = depth_flat > 0

    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    depth_valid = depth_flat[valid]

    # Calcular las coordenadas 3D en el sistema de la cámara
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid

    points_camera = np.vstack((x, y, z)).T  # Forma: (N, 3)

    # Transformar los puntos al sistema de coordenadas del mundo
    R_WC = T_WC[:3, :3]
    t_WC = T_WC[:3, 3]
    points_world = (R_WC @ points_camera.T).T + t_WC  # Forma: (N, 3)

    return points_world

def validate(flags):
    absolute_path = os.path.abspath(flags.path)
    if not os.path.exists(os.path.join(flags.path, 'odometry.csv')):
        print(f"El directorio {absolute_path} no contiene 'odometry.csv'.")
        return False
    if not os.path.exists(os.path.join(flags.path, 'depth')):
        print(f"El directorio {absolute_path} no contiene la carpeta 'depth'.")
        return False
    return True

def main():
    flags = read_args()
    if not validate(flags):
        return

    data = read_data(flags)
    viewer = PangolinViewer()
    trajectory_points = []
    point_cloud_points = []
    trajectory = [] 

    for i in range(0, len(data['poses']), flags.every):
        T_WC = data['poses'][i]
        # Posición de la cámara
        position = T_WC[:3, 3]
        trajectory_points.append(position)
        trajectory_np = np.array(trajectory_points)
        trajectory.append(T_WC.copy()) 

        # Computar la nube de puntos para este frame
        points_world = compute_point_cloud_for_frame(i, data)

        if points_world is not None and len(points_world) > 0:
            point_cloud_points.append(points_world)

        # Concatenar todos los puntos
        all_points = np.vstack(point_cloud_points) if point_cloud_points else None

        # Submuestrear la nube de puntos para visualización
        if all_points is not None and len(all_points) > 100000:
            idx = np.random.choice(len(all_points), 100000, replace=False)
            all_points_sampled = all_points[idx]
        else:
            all_points_sampled = all_points

        # Actualizar el visualizador
        viewer.update(trajectory_np, all_points_sampled)

        # Controlar la velocidad de actualización
        time.sleep(0.05)

    trajectory_array = np.array(trajectory)
    np.save('slam_lidar.npy', trajectory_array)
    print("Trayectoria guardada en 'slam_lidar.npy'")
    

if __name__ == "__main__":
    main()
