import numpy as np
import pangolin
import OpenGL.GL as gl
import sys
import os

def draw_trajectory(trajectory, window_title):
    pangolin.CreateWindowAndBind(window_title, 1024, 768)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Configurar la cámara
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin.ModelViewLookAt(0, -10, -20,
                                 0, 0, 0,
                                 0, -1, 0)
    )

    handler = pangolin.Handler3D(scam)

    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0)
    dcam.SetHandler(handler)

    # Extraer las posiciones de la trayectoria
    positions = trajectory[:, :3, 3]

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        dcam.Activate(scam)

        # Dibujar la trayectoria
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 1.0, 0.0)  # Color verde para la trayectoria
        pangolin.DrawLine(positions)

        # Marcar el punto de inicio
        gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)  # Color rojo para el inicio
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(positions[0][0], positions[0][1], positions[0][2])
        gl.glEnd()

        # Marcar el punto de finalización
        gl.glPointSize(10)
        gl.glColor3f(0.0, 0.0, 1.0)  # Color azul para el final
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(positions[-1][0], positions[-1][1], positions[-1][2])
        gl.glEnd()

        pangolin.FinishFrame()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        
        sys.exit(-1)

    # Cargar la trayectoria guardada
    trajectory_file = sys.argv[1]
    trajectory_array = np.load(trajectory_file)

    # Determinar el título de la ventana según el archivo
    filename = os.path.basename(trajectory_file)
    print(f"File is: {filename}")
    if 'lidar' in filename.lower():
        window_title = 'LIDAR SLAM'
    elif 'monocular' in filename.lower():
        window_title = 'MONOCULAR SLAM'
    else:
        window_title = 'Trajectory Viewer'

    draw_trajectory(trajectory_array, window_title)
