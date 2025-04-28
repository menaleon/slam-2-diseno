import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Diccionario: nombre -> archivo CSV
trayectorias = {
    "VisualSLAM3D": "/home/jimena/Escritorio/PROYECTO/REPO/COMPARATOR/trayectoria_sin_bundle_adjustment.csv",
    "BundleAdjustmentSLAM": "/home/jimena/Escritorio/PROYECTO/REPO/COMPARATOR/trayectoria_con_bundle_adjustment.csv",
    "KeyframeSLAM": "/home/jimena/Escritorio/PROYECTO/REPO/Keyframes/trayectoria_keyframe_visualslam.csv",
    "PoseGraphSLAM": "/home/jimena/Escritorio/PROYECTO/REPO/posegraph/trayectoria_pose_graph_slam.csv"
}

# Colores para graficar
colores = {
    "VisualSLAM3D": "orange",
    "BundleAdjustmentSLAM": "blue",
    "KeyframeSLAM": "green",
    "PoseGraphSLAM": "red"
}

plt.figure(figsize=(10, 7))

for nombre, archivo in trayectorias.items():
    if os.path.exists(archivo):
        data = pd.read_csv(archivo)
        x = data["X"].values
        z = data["Z"].values

        # Distancia total recorrida (suma de desplazamientos)
        dist_total = np.sum(np.linalg.norm(np.diff(np.vstack((x, z)).T, axis=0), axis=1))

        # Desplazamiento directo desde inicio a fin
        desplazamiento = np.linalg.norm([x[-1] - x[0], z[-1] - z[0]])

        # Graficar
        plt.plot(x, z, label=f"{nombre} (Dist: {dist_total:.1f}, Δ: {desplazamiento:.1f})", color=colores[nombre])
        plt.scatter(x[0], z[0], marker='o', color=colores[nombre])
        plt.scatter(x[-1], z[-1], marker='x', color=colores[nombre])
    else:
        print(f"[!] Archivo no encontrado: {archivo}")

plt.title("Comparación de Trayectorias SLAM 2D")
plt.xlabel("X Position")
plt.ylabel("Z Position")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.savefig("comparacion_trayectorias.png")
plt.show()
