from multiprocessing import Process, Queue
import numpy as np
import OpenGL.GL as gl
import pypangolin
import g2o


"""
def optimize(frames, points, local_window=10, fix_points=True, verbose=True, rounds=10):
    
    Optimización simple de poses y puntos clave (Bundle Adjustment simplificado).
    
    err = 0.0
    for _ in range(rounds):
        for f in frames[-local_window:]:  # Solo optimiza las últimas `local_window` frames
            for p in points:
                # Calcular el error de reproyección para cada punto
                reprojection_errs = []
                for frame, idx in zip(p.frames, p.idxs):
                    uv = frame.key_pts[idx]  # Keypoints observados en 2D
                    proj = np.dot(frame.pose[:3], np.append(p.pt[:3], 1))  # Proyección del punto 3D en coordenadas homogéneas
                    proj = proj[0:2] / proj[2]  # Normalizar para obtener coordenadas 2D
                    reprojection_errs.append(np.linalg.norm(proj - uv))  # Error de reproyección

                # Sumar el error de reproyección
                err += np.mean(reprojection_errs)

        # Mostrar el error en cada iteración si verbose está activado
        if verbose:
            print(f"Iteración de optimización completada, error: {err}")

    return err

"""
class Point(object):
  # A Point is a 3-D point in the world
  # Each Point is observed in multiple Frames

  def __init__(self, mapp, loc):
    self.pt = loc # 3D location of homogeneous coordinates 
    self.frames = [] # List of frames where this point was observed
    self.idxs = [] # Id's of locations of this points in its frames
    
    self.id = len(mapp.points) 
    mapp.points.append(self)

  def add_observation(self, frame, idx):
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)

class Descriptor(object):
  def __init__(self):
    self.frames = []
    self.points = []
    self.state = None
    self.q = None
    self.local_window = 10  # Número de fotogramas en la ventana local
    self.fix_points = True  # Fijar algunos puntos como referencia
    self.verbose = True  # Activar el modo detallado
    self.rounds = 10  # Número de iteraciones de optimización
    self.CULLING_ERR_THRES = 1.0
    #self.max_frame = 0
  # G2O optimization:
  """def optimize(self):
    err = optimize(self.frames, self.points, self.local_window, self.fix_points, self.verbose, self.rounds)

    # Poda de puntos clave (Key-Point Pruning)
    culled_pt_count = 0
    for p in self.points:
        # Punto viejo si ha sido observado en menos de 4 fotogramas o si es muy antiguo
        old_point = len(p.frames) <= 4 and p.frames[-1].id + 7 < self.max_frame

        # Manejo del error de reproyección
        errs = []
        for f, idx in zip(p.frames, p.idxs):
            uv = f.key_pts[idx]  # Asegúrate de que "key_pts" es el nombre correcto
            proj = np.dot(f.pose[:3], np.append(p.pt[:3], 1)) # Proyección del punto 3D al espacio de imagen
            proj = proj[0:2] / proj[2]  # Normalización
            errs.append(np.linalg.norm(proj - uv))  # Error de reproyección

        # Si el punto es "viejo" o el error promedio es mayor que el umbral, eliminarlo
        if old_point or np.mean(errs) > self.CULLING_ERR_THRES:
            culled_pt_count += 1
            self.points.remove(p)  # Remover el punto de la lista
            #p.delete()  # Llamar al método de eliminación del punto

    if self.verbose:
        print(f"Eliminados {culled_pt_count} puntos clave con errores altos")

    # Retornar el error de optimización
    return err """

  def create_viewer(self):
    self.q = Queue()
    self.vp = Process(target=self.viewer_thread, args=(self.q,))
    self.vp.daemon = True
    self.vp.start()

  def viewer_thread(self, q):
    self.viewer_init(1024, 768)
    while 1:
      self.viewer_refresh(q)

  def viewer_init(self, w, h):
    pypangolin.CreateWindowAndBind('Main', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pypangolin.OpenGlRenderState(
      pypangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
      pypangolin.ModelViewLookAt(0, -10, -8,
                               0, 0, 0,
                               0, -1, 0))
    self.handler = pypangolin.Handler3D(self.scam)

    # Create Interactive View in window
    self.dcam = pypangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
    self.dcam.SetHandler(self.handler)

  def viewer_refresh(self, q):
    if self.state is None or not q.empty():
      self.state = q.get()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0, 0, 0, 0)
    self.dcam.Activate(self.scam)
    
    # Draw Point Cloud
    #points = np.random.random((10000, 3))
    #colors = np.zeros((len(points), 3))
    #colors[:, 1] = 1 -points[:, 0]
    #colors[:, 2] = 1 - points[:, 1]
    #colors[:, 0] = 1 - points[:, 2]
    #points = points * 3 + 1
    #gl.glPointSize(10)
    #pypangolin.DrawPoints(self.state[1], colors)

    # draw keypoints
    gl.glPointSize(2)
    gl.glColor3f(0.184314, 0.309804, 0.184314)
    pypangolin.DrawPoints(self.state[1]+1)
    gl.glPointSize(1)
    gl.glColor3f(0.3099, 0.3099,0.184314)
    pypangolin.DrawPoints(self.state[1])

    # draw poses
    gl.glColor3f(0.0, 1.0, 1.0)
    pypangolin.DrawCameras(self.state[0])

    pypangolin.FinishFrame()

  def display(self):
    if self.q is None:
      return
    poses, pts = [], []
    for f in self.frames:
      poses.append(f.pose)
    for p in self.points:
      pts.append(p.pt)
    self.q.put((np.array(poses), np.array(pts)))


  def optimize(self, max_iterations=10):
        # Creación del optimizador g2o
        optimizer = g2o.SparseOptimizer()

        # Usar LinearSolverDenseSE3 para el solver
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)

        # Crear los parámetros de la cámara antes de agregar vértices y edges
        fx, fy = self.frames[0].count[0, 0], self.frames[0].count[1, 1]
        cx, cy = self.frames[0].count[0, 2], self.frames[0].count[1, 2]

        # Parámetros de la cámara (distorsión se establece en 0 si no se usa)
        camera_params = g2o.CameraParameters(fx, np.array([cx, cy]), 0)  

        # Asignar un ID y agregar los parámetros al optimizador
        camera_params.set_id(0)
        optimizer.add_parameter(camera_params)


        # Agregar poses de las cámaras como vértices en el gráfico
        for frame in self.frames:
          v_se3 = g2o.VertexSE3()
          v_se3.set_id(frame.id)

          # Extraer la rotación y la traslación de la pose
          rotation = frame.pose[:3, :3]  # Matriz de rotación 3x3
          translation = frame.pose[:3, 3]  # Vector de traslación 3D

          # Crear el objeto Isometry3d manualmente
          isometry = g2o.Isometry3d()
          isometry.set_translation(translation)  # Establecer la traslación
          isometry.set_rotation(rotation)  # Establecer la rotación

          v_se3.set_estimate(isometry)  # Establecer la pose en el vértice
          v_se3.set_fixed(frame.id == 0)  # Fijar la primera pose
          optimizer.add_vertex(v_se3)


        # Agregar puntos 3D como vértices en el gráfico
        point_id_offset = len(self.frames)
        for i, point in enumerate(self.points):
            v_p = g2o.VertexPointXYZ()
            v_p.set_id(point_id_offset + i)
            v_p.set_estimate(point.pt[:3])
            v_p.set_marginalized(True)
            optimizer.add_vertex(v_p)


            # Agregar restricciones (observaciones) para cada punto
            for frame, idx in zip(point.frames, point.idxs):
              edge = g2o.EdgeProjectXYZ2UV()
              edge.set_vertex(0, v_p)  # Punto 3D
              edge.set_vertex(1, optimizer.vertex(frame.id))  # Cámara
              edge.set_measurement(frame.key_pts[idx])  # Medida 2D
              edge.set_information(np.eye(2))  # Matriz de información (confianza)

              # Asignar los parámetros de la cámara al edge
              edge.set_parameter_id(0, 0)
              
              edge.set_robust_kernel(g2o.RobustKernelHuber())  # Opcional: añade robustez a la estimación
              optimizer.add_edge(edge)


        # Ejecutar la optimización
        optimizer.initialize_optimization()
        optimizer.optimize(max_iterations)

        
