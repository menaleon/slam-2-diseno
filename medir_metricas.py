import time
import psutil
import subprocess
import json
import os
from datetime import datetime

"""
medir_metricas(comando, nombre_slam=None): mide recursos del proceso.
Si nombre_slam se especifica, guarda los resultados como metricas.json
en la carpeta resultados/<nombre_slam>/<timestamp>/
"""

def medir_metricas(comando, nombre_slam=None):
    tiempo_inicio = time.time()
    proceso = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps_proceso = psutil.Process(proceso.pid)

    uso_cpu_usuario = 0
    uso_cpu_kernel = 0
    max_memoria = 0

    try:
        while proceso.poll() is None:
            cpu_times = ps_proceso.cpu_times()
            uso_cpu_usuario = cpu_times.user
            uso_cpu_kernel = cpu_times.system
            mem_info = ps_proceso.memory_info()
            max_memoria = max(max_memoria, mem_info.rss)
            time.sleep(0.1)

        stdout, stderr = proceso.communicate()
    except psutil.NoSuchProcess:
        stdout, stderr = b"", b"Proceso no disponible"

    tiempo_total = time.time() - tiempo_inicio

    resultados = {
        "tiempo_total_seg": round(tiempo_total, 3),
        "cpu_user_seg": round(uso_cpu_usuario, 4),
        "cpu_kernel_seg": round(uso_cpu_kernel, 4),
        "memoria_max_kb": round(max_memoria / 1024, 2),
        "codigo_retorno": proceso.returncode,
        "salida": stdout.decode(errors="ignore"),
        "errores": stderr.decode(errors="ignore")
    }

    if nombre_slam:
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        output_dir = os.path.join("resultados", nombre_slam, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "metricas.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)

    return resultados
