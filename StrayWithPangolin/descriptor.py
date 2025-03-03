import numpy as np
import pypangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue
import time

class PangolinViewer:
    def __init__(self):
        self.state = None
        self.q = Queue()
        self.viewer_process = Process(target=self.viewer_thread, args=(self.q,))
        self.viewer_process.daemon = True
        self.viewer_process.start()

    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while not pypangolin.ShouldQuit():
            self.viewer_refresh(q)
            time.sleep(0.01)  # Evita alto uso de CPU

    def viewer_init(self, w, h):
        pypangolin.CreateWindowAndBind('SLAM Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pypangolin.OpenGlRenderState(
            pypangolin.ProjectionMatrix(w, h, 500, 500, w//2, h//2, 0.1, 1000),
            pypangolin.ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0)
        )

        self.handler = pypangolin.Handler3D(self.scam)
        self.dcam = pypangolin.CreateDisplay().SetBounds(
            pypangolin.Attach(0), pypangolin.Attach(1),
            pypangolin.Attach(0), pypangolin.Attach(1), -w/h
        ).SetHandler(self.handler)

    def viewer_refresh(self, q):
        while not q.empty():
            self.state = q.get()
        if self.state is None:
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        # Dibujar la trayectoria de la cámara
        if 'trajectory' in self.state:
            trajectory = self.state['trajectory']
            gl.glColor3f(1.0, 0.0, 0.0)
            pypangolin.DrawLine(trajectory)

        # Dibujar la nube de puntos
        if 'points' in self.state and self.state['points'] is not None:
            points = self.state['points']
            gl.glPointSize(2)
            gl.glColor3f(0.0, 1.0, 0.0)
            pypangolin.DrawPoints(points)

        pypangolin.FinishFrame()

    def update(self, trajectory, points=None):
        state = {'trajectory': trajectory}
        if points is not None:
            state['points'] = points
        self.q.put(state)
