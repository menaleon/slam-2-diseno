import numpy as np
import pypangolin
import OpenGL.GL as gl
import sys
import os

def draw_trajectory(trajectory, window_title):
    pypangolin.CreateWindowAndBind(window_title, 1024, 768)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Configurar la cámara
    scam = pypangolin.OpenGlRenderState(
        pypangolin.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pypangolin.ModelViewLookAt(0, -10, -20,
                                 0, 0, 0,
                                 0, -1, 0)
    )

    handler = pypangolin.Handler3D(scam)

    dcam = pypangolin.CreateDisplay()
    dcam.SetBounds(
        pypangolin.Attach.Pix(0),
        pypangolin.Attach.Pix(1),
        pypangolin.Attach.Pix(0),
        pypangolin.Attach.Pix(1)
    )    
    dcam.SetHandler(handler)

    positions = trajectory[:, :3, 3]

    while not pypangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        dcam.Activate(scam)

        gl.glLineWidth(2)
        gl.glColor3f(0.0, 1.0, 0.0) 
        pypangolin.DrawLine(positions)

        # for start point
        gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)  # red for start
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(positions[0][0], positions[0][1], positions[0][2])
        gl.glEnd()

        # for end point
        gl.glPointSize(10)
        gl.glColor3f(0.0, 0.0, 1.0)  # blue for end
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(positions[-1][0], positions[-1][1], positions[-1][2])
        gl.glEnd()

        pypangolin.FinishFrame()

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
