import os
import subprocess
import tkinter as tk
from tkinter import messagebox

# Carpeta raiz
CARPETA_LMS = "LMS"

#Obtiene los nombres de subcarpetas en la carpeta LMS que empiecen con LMS_
def obtener_tipos_lms():
    if not os.path.exists(CARPETA_LMS):
        return []
    return [nombre for nombre in os.listdir(CARPETA_LMS)
            if os.path.isdir(os.path.join(CARPETA_LMS, nombre)) and nombre.startswith("LMS_")]


def ejecutar_lms(nombre_lms):
    messagebox.showinfo("LMS seleccionado", f"Ejecutando: {nombre_lms}")
    
    try:
        # Esta línea espera a que el script termine antes de continuar
        subprocess.run(["python3", f"{CARPETA_LMS}/{nombre_lms}/main2.py"], check=True)
        messagebox.showinfo("Éxito", f"{CARPETA_LMS}/{nombre_lms}/{nombre_lms}.py finalizó correctamente.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error en la ejecución de {CARPETA_LMS}/{nombre_lms}/{nombre_lms}.py:\n{e}")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Archivo {CARPETA_LMS}/{nombre_lms}/{nombre_lms}.py no encontrado")
    

def calcular_metricas():
    messagebox.showinfo("Métricas", "Calculando métricas de rendimiento...")


# Ventana principal
ventana = tk.Tk()
ventana.title("Plataforma LMS múltiple")
ventana.configure(bg="#ffff99")  # Fondo amarillo claro

# Etiquetas superiores
tk.Label(ventana, text="TEC | 2025", font=("Arial", 12, "bold"), bg="#ffff99").pack(pady=(10, 0))
tk.Label(ventana, text="Plataforma LMS múltiple", font=("Arial", 18, "bold"), bg="#ffff99").pack(pady=(5, 5))
tk.Label(ventana, text="Experimente con algunas opciones de\nSLAM (LMS). Utilice los botones para\ngenerar un mapa en 2D a partir de un video .mp4",
         font=("Arial", 10), bg="#ffff99").pack()

# Contenedor para los botones de LMS
frame_botones = tk.Frame(ventana, bg="#ffff99")
frame_botones.pack(pady=20)

# Botones de LMS y enlazamiento con funciones
tipos_lms = obtener_tipos_lms()
for i, lms in enumerate(tipos_lms):
    nombre_mostrado = lms.replace("LMS_", "LMS tipo ")
    boton = tk.Button(frame_botones, text=nombre_mostrado, font=("Arial", 12, "bold"),
                      bg="#66ff66", relief="raised", width=19,
                      command=lambda lms=lms: ejecutar_lms(lms))
    boton.grid(row=0, column=i, padx=10)

# Boton de metricas
tk.Button(ventana, text="Calcular métricas\nde rendimiento", font=("Arial", 12, "bold"),
          bg="#9999ff", fg="black", relief="raised", width=25, height=2,
          command=calcular_metricas).pack(pady=10)

# Ejecutar la ventana
ventana.mainloop()
