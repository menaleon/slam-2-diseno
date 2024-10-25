import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import os
import subprocess
import threading

# Variables globales para root y style
root = None
style = None

def run_monocular_slam():
    dataset_path = filedialog.askdirectory(title="Selecciona la carpeta del conjunto de datos")
    if dataset_path:
        rgb_video = os.path.join(dataset_path, 'rgb.mp4')
        if os.path.exists(rgb_video):
            threading.Thread(target=execute_monocular_slam, args=(dataset_path,)).start()
        else:
            messagebox.showerror("Error", "La carpeta seleccionada no contiene 'rgb.mp4'.")

def execute_monocular_slam(dataset_path):
    try:
        subprocess.run(['python3', '../SLAMPy-Monocular-SLAM-implementation-in-python/slam.py', dataset_path], check=True)
        messagebox.showinfo("Éxito", "Mapeo terminado exitosamente.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error al ejecutar SLAM monocular:\n{e}")

def run_lidar_slam():
    dataset_path = filedialog.askdirectory(title="Selecciona la carpeta del conjunto de datos")
    if dataset_path:
        odometry_file = os.path.join(dataset_path, 'odometry.csv')
        depth_dir = os.path.join(dataset_path, 'depth')
        if os.path.exists(odometry_file) and os.path.exists(depth_dir):
            threading.Thread(target=execute_lidar_slam, args=(dataset_path,)).start()
        else:
            messagebox.showerror("Error", "La carpeta seleccionada no contiene los archivos necesarios para SLAM Lidar.")

def execute_lidar_slam(dataset_path):
    try:
        subprocess.run(['python3', '../StrayWithPangolin/main.py', dataset_path], check=True)
        messagebox.showinfo("Éxito", "Mapeo terminado exitosamente.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error al ejecutar SLAM Lidar:\n{e}")

def compare_results():
    monocular_trajectory = 'slam_monocular.npy'
    lidar_trajectory = 'slam_lidar.npy'
    missing = []
    if not os.path.exists(monocular_trajectory):
        missing.append('monocular')
    if not os.path.exists(lidar_trajectory):
        missing.append('lidar')

    if missing:
        algos = ' y '.join(missing)
        messagebox.showerror("Error", f"Debes ejecutar el algoritmo {algos} para comparar los resultados.")
    else:
        # Ejecutar visualize_trajectory.py para ambas trayectorias y mostrar la ventana de leyenda
        execute_visualization_with_legend(monocular_trajectory, lidar_trajectory)

def execute_visualization_with_legend(monocular_trajectory, lidar_trajectory):
    # Crear una lista para almacenar los procesos de visualización
    pangolin_processes = []

    # Iniciar las visualizaciones en procesos separados
    def start_visualization(trajectory_file):
        process = subprocess.Popen(['python3', 'visualize_trajectory.py', trajectory_file])
        pangolin_processes.append(process)

    threading.Thread(target=start_visualization, args=(monocular_trajectory,)).start()
    threading.Thread(target=start_visualization, args=(lidar_trajectory,)).start()

    # Mostrar la ventana de leyenda
    show_legend_window(pangolin_processes)

def show_legend_window(pangolin_processes):
    legend_window = ttk.Toplevel()
    legend_window.title("Leyenda")
    legend_window.geometry("300x200")
    legend_window.configure(background=style.colors.bg)

    # Crear un marco para centrar los widgets
    frame = ttk.Frame(legend_window, padding=10)
    frame.pack(expand=True, fill=BOTH)

    # Punto rojo y etiqueta
    red_dot = ttk.Canvas(frame, width=20, height=20, background=style.colors.bg)
    red_dot.create_oval(2, 2, 18, 18, fill='red')
    red_dot.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    red_label = ttk.Label(frame, text="Indica punto de inicio", foreground='white')
    red_label.grid(row=0, column=1, sticky='w')

    # Punto azul y etiqueta
    blue_dot = ttk.Canvas(frame, width=20, height=20, background=style.colors.bg)
    blue_dot.create_oval(2, 2, 18, 18, fill='blue')
    blue_dot.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    blue_label = ttk.Label(frame, text="Indica punto de final", foreground='white')
    blue_label.grid(row=1, column=1, sticky='w')

    # Botón "Ok" para cerrar las ventanas
    def on_ok():
        # Cerrar los procesos de Pangolin
        for process in pangolin_processes:
            process.terminate()
        # Cerrar la ventana de leyenda
        legend_window.destroy()

    # Botón con estilo personalizado
    ok_button = ttk.Button(frame, text="Ok", command=on_ok, width=15, style='Custom.TButton')
    ok_button.grid(row=2, column=0, columnspan=2, pady=20)

    # Configurar para que la ventana de leyenda sea modal
    legend_window.transient(root)
    legend_window.grab_set()
    root.wait_window(legend_window)

def main():
    global root, style  # Asegurar que root y style son accesibles globalmente
    # Inicializar ttkbootstrap con un tema oscuro
    style = ttk.Style(theme="darkly")
    root = style.master
    root.title("SLAM")
    root.geometry("700x300")
    root.resizable(False, False)
    root.configure(background=style.colors.bg)

    # Crear un marco para centrar los widgets
    frame = ttk.Frame(root)
    frame.place(relx=0.5, rely=0.5, anchor='center')

    label = ttk.Label(frame, text="Seleccione una opción:", font=("Helvetica", 14), foreground='white')
    label.pack(pady=10)

    # Estilo personalizado para los botones con sombra simulada
    style.configure('Custom.TButton', font=('Helvetica', 12), padding=10)
    style.map('Custom.TButton',
              foreground=[('!disabled', 'white')],
              background=[('!disabled', style.colors.primary)],
              relief=[('pressed', 'sunken'), ('!pressed', 'raised')])

    # Botones con estilo personalizado
    btn_monocular = ttk.Button(frame, text="Ejecutar SLAM Monocular", command=run_monocular_slam, width=30, style='Custom.TButton')
    btn_monocular.pack(pady=5)

    btn_lidar = ttk.Button(frame, text="Ejecutar SLAM Lidar", command=run_lidar_slam, width=30, style='Custom.TButton')
    btn_lidar.pack(pady=5)

    btn_compare = ttk.Button(frame, text="Comparar resultados", command=compare_results, width=30, style='Custom.TButton')
    btn_compare.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
