import os
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
import open3d as o3d
import time

# Descripción del script
description = """
Este script visualiza datasets tomados con la aplicación Stray Scanner usando Open3D.
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
    return {'poses': poses, 'intrinsics': intrinsics, 'depth_frames': depth_frames}

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
    u, v, depth = u.flatten(), v.flatten(), depth.flatten()
    valid = depth > 0
    
    x = (u[valid] - cx) * depth[valid] / fx
    y = (v[valid] - cy) * depth[valid] / fy
    z = depth[valid]
    points_camera = np.vstack((x, y, z)).T
    
    R_WC, t_WC = T_WC[:3, :3], T_WC[:3, 3]
    points_world = (R_WC @ points_camera.T).T + t_WC
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
    trajectory_points = []
    point_cloud_points = []
    trajectory = [] 
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    point_cloud = o3d.geometry.PointCloud()
    traj_line_set = o3d.geometry.LineSet()
    
    for i in range(0, len(data['poses']), flags.every):
        T_WC = data['poses'][i]
        position = T_WC[:3, 3]
        trajectory_points.append(position)
        trajectory.append(T_WC.copy())
        
        points_world = compute_point_cloud_for_frame(i, data)
        if points_world is not None and len(points_world) > 0:
            point_cloud_points.append(points_world)
        
        all_points = np.vstack(point_cloud_points) if point_cloud_points else None
        if all_points is not None and len(all_points) > 100000:
            idx = np.random.choice(len(all_points), 100000, replace=False)
            all_points_sampled = all_points[idx]
        else:
            all_points_sampled = all_points
        
        if all_points_sampled is not None:
            point_cloud.points = o3d.utility.Vector3dVector(all_points_sampled)
            point_cloud.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (len(all_points_sampled), 1)))
            vis.add_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()
        
        time.sleep(0.05)
    
    trajectory_array = np.array(trajectory)
    np.save('slam_lidar.npy', trajectory_array)
    print("Trayectoria guardada en 'slam_lidar.npy'")
    vis.run()
    vis.destroy_window()
    
if __name__ == "__main__":
    main()
