import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
import pangolin
import OpenGL.GL as gl

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0

def read_args():
    parser = ArgumentParser(description="Visualize trajectory using Pangolin.")
    parser.add_argument('path', type=str, help="Path to StrayScanner dataset to process.")
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
    return { 'poses': poses, 'intrinsics': intrinsics }

def scale_poses(poses, scale_factor=1.0):
    scaled_poses = []
    for pose in poses:
        pose[:3, 3] *= scale_factor  # Escalar solo las posiciones
        scaled_poses.append(pose)
    return scaled_poses

def initialize_pangolin(poses):
    # Calcular el centro de la trayectoria para centrar la cámara en esa región
    center = np.mean([pose[:3, 3] for pose in poses], axis=0)
    print(f"Centro calculado de la trayectoria: {center}")  # Agregar print para verificar el centro

    pangolin.CreateWindowAndBind('Trajectory Viewer', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Centrar la cámara en la región de la trayectoria
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000),
        pangolin.ModelViewLookAt(center[0], center[1], center[2] + 10, center[0], center[1], center[2], 0, -1, 0)
    )

    handler = pangolin.Handler3D(scam)
    dcam = pangolin.CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -640/480).SetHandler(handler)
    return dcam, scam

def draw_trajectory(poses):
    # Limpiar los buffers de color y profundidad
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    for i in range(min(5, len(poses))):
        print(f"Posición {i}: {poses[i][:3, 3]}")
    for i in range(1, len(poses)):
        # Extraer las posiciones de las matrices de transformación
        p1 = poses[i-1][:3, 3]
        p2 = poses[i][:3, 3]

        # Dibujar la línea entre las posiciones p1 y p2
        gl.glLineWidth(2)
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1.0, 0.0, 0.0)  # Color rojo para las líneas
        gl.glVertex3f(p1[0], p1[1], p1[2])
        gl.glVertex3f(p2[0], p2[1], p2[2])
        gl.glEnd()

        # Dibujar un punto en cada posición
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(0.0, 1.0, 0.0)  # Color verde para los puntos
        gl.glVertex3f(p1[0], p1[1], p1[2])
        gl.glEnd()

    # Dibujar un punto grande en la primera posición de la trayectoria
    gl.glPointSize(10)
    gl.glBegin(gl.GL_POINTS)
    gl.glColor3f(1.0, 1.0, 0.0)  # Color amarillo para resaltar
    gl.glVertex3f(poses[0][:3, 3][0], poses[0][:3, 3][1], poses[0][:3, 3][2])
    gl.glEnd()

    # Finalizar el frame
    pangolin.FinishFrame()

def main():
    flags = read_args()

    if not os.path.exists(os.path.join(flags.path, 'rgb.mp4')):
        print(f"Invalid dataset path: {flags.path}")
        return

    # Leer los datos de odometría y las poses
    data = read_data(flags)

    # Escalar las poses si es necesario
    # data['poses'] = scale_poses(data['poses'], scale_factor=1.0)

    # Inicializar la ventana de Pangolin
    dcam, scam = initialize_pangolin(data['poses'])

    while not pangolin.ShouldQuit():
        # Actualizar la visualización de Pangolin con las poses
        draw_trajectory(data['poses'])

if __name__ == "__main__":
    main()
