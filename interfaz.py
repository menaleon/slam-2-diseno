import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from medir_metricas import medir_metricas

CARPETA_LMS = "LMS"

def obtener_tipos_lms():
    return [
        nombre for nombre in os.listdir(CARPETA_LMS)
        if os.path.isdir(os.path.join(CARPETA_LMS, nombre)) and nombre.startswith("LMS_")
    ] if os.path.exists(CARPETA_LMS) else []

def seleccionar_video():
    return filedialog.askopenfilename(
        title="Seleccionar video .mp4",
        filetypes=[("Archivos de video", "*.mp4")]
    )

def ejecutar_lms(nombre_lms):
    ruta_video = seleccionar_video()
    if not ruta_video:
        messagebox.showwarning("Selección cancelada", "No se seleccionó ningún video.")
        return

    messagebox.showinfo("LMS seleccionado", f"Ejecutando: {nombre_lms}\nCon video:\n{ruta_video}")
    script_path = os.path.join(CARPETA_LMS, nombre_lms, "posegraph.py")
    comando = ["python3", script_path, ruta_video]

    resultados = medir_metricas(comando)

    salida_filtrada = {}
    for clave, valor in resultados.items():
        if isinstance(valor, str):
            if not valor.strip():
                continue
            if len(valor) > 500:
                salida_filtrada[clave] = valor[:500] + "... (truncado)"
                continue
        salida_filtrada[clave] = valor

    mensaje = json.dumps(salida_filtrada, indent=4, ensure_ascii=False)
    messagebox.showinfo("Resultados de ejecución", mensaje)

def crear_interfaz():
    ventana = tk.Tk()
    ventana.title("Plataforma LMS múltiple")
    ventana.geometry("800x500")
    ventana.resizable(False, False)

    # === Canvas con degradado ===
    canvas = tk.Canvas(ventana, width=800, height=500)
    canvas.pack(fill="both", expand=True)

    # Degradado de #99c2ff a #f0f4f8
    for i in range(500):
        r = int(153 + (240 - 153) * (i / 500))  # Red: 153 → 240
        g = int(194 + (244 - 194) * (i / 500))  # Green: 194 → 244
        b = int(255 + (248 - 255) * (i / 500))  # Blue: 255 → 248
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, i, 800, i, fill=color)

    estilo = ttk.Style()
    estilo.theme_use('clam')
    estilo.configure("TButton",
                     font=("Segoe UI", 11, "bold"),
                     padding=10,
                     background="#388e3c",
                     foreground="white",
                     relief="flat")
    estilo.map("TButton", background=[("active", "#2e7d32")])
    estilo.configure("TLabel", background="#ffffff", foreground="#333")

    frame = tk.Frame(canvas, bg="white", bd=2, relief="ridge")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=350)

    tk.Label(frame, text="Plataforma LMS múltiple", font=("Helvetica Neue", 20, "bold"), bg="white", fg="#0d47a1").pack(pady=(20, 10))
    tk.Label(frame, text="Seleccione un LMS y un video .mp4 para generar una trayectoria 2D", font=("Segoe UI", 11), bg="white").pack()

    # === Botones ===
    frame_botones = tk.Frame(frame, bg="white")
    frame_botones.pack(pady=20)

    tipos_lms = obtener_tipos_lms()
    for i, lms in enumerate(tipos_lms):
        nombre_mostrado = lms.replace("LMS_", "LMS tipo ")
        boton = ttk.Button(frame_botones, text=nombre_mostrado, command=lambda lms=lms: ejecutar_lms(lms))
        boton.grid(row=i // 2, column=i % 2, padx=15, pady=10)

    tk.Label(frame, text="© TEC 2025 | Proyecto SLAM", font=("Segoe UI", 11), bg="white", fg="black").pack(side="bottom", pady=10)

    ventana.mainloop()

if __name__ == "__main__":
    crear_interfaz()
