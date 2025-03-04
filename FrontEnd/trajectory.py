import numpy as np
import pypangolin
import OpenGL.GL as gl

import pypangolin
import OpenGL.GL as gl
import numpy as np

def draw_multiple_trajectories(trajectories, colors, titles, window_title="Trajectories Viewer"):
    """Dibuja múltiples trayectorias en una sola ventana de pypangolin."""
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
        pypangolin.Attach.Pix(768),  # Altura de la ventana
        pypangolin.Attach.Pix(0),
        pypangolin.Attach.Pix(1024)  # Ancho de la ventana
    )
    dcam.SetHandler(handler)

    while not pypangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        dcam.Activate(scam)

        for trajectory, color, title in zip(trajectories, colors, titles):
            positions = trajectory[:, :3, 3].astype(np.float32)  # Convertir a float32 para glDrawLines
            
            # Dibujar la línea de la trayectoria con `glDrawLines`
            gl.glLineWidth(2)
            gl.glColor3f(*color)
            pypangolin.glDrawLines(positions)

            # Punto de inicio
            gl.glPointSize(10)
            gl.glColor3f(1.0, 0.0, 0.0)  # Rojo para el inicio
            gl.glBegin(gl.GL_POINTS)
            gl.glVertex3f(*positions[0])
            gl.glEnd()

            # Punto de fin
            gl.glPointSize(10)
            gl.glColor3f(0.0, 0.0, 1.0)  # Azul para el fin
            gl.glBegin(gl.GL_POINTS)
            gl.glVertex3f(*positions[-1])
            gl.glEnd()

        pypangolin.FinishFrame()

def visualize_trajectories(trajectories_to_load):
    # Listas para trayectorias, colores y títulos
    trajectories = []
    colors = []
    titles = []

    # Colores para las trayectorias
    color_map = {
        "monocular": (0.0, 1.0, 0.0),  # Verde
        "sift": (1.0, 0.5, 0.0),       # Naranja
        "optimize": (0.5, 0.0, 1.0)    # Morado
    }

    # Cargar las trayectorias especificadas
    for name, path in trajectories_to_load:
        trajectory = np.load(path)
        trajectories.append(trajectory)
        colors.append(color_map.get(name, (1.0, 1.0, 1.0)))  # Blanco por defecto si no hay color
        titles.append(name.upper())

    # Dibujar todas las trayectorias en una ventana
    draw_multiple_trajectories(trajectories, colors, titles)

if __name__ == "__main__":
    # Especificar trayectorias para cargar
    trajectories_to_load = [
        #("monocular", "slam_monocular.npy"),
        ("sift", "slam_sift.npy"),
        #("optimize", "slam_optimize.npy")
    ]
    # Puedes comentar alguna de las líneas anteriores para cargar solo las trayectorias que desees

    # Ejecutar la visualización
    visualize_trajectories(trajectories_to_load)
