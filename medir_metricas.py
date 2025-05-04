import time
import psutil
import subprocess

"""
medir_metricas(comando): se encarga de contabilizar metricas de memoria y tiempos relevantes
                         sobre los tipos de LMS

Entrada: comando (la ruta del archivo LMS por ejecutar)
Salida: un diccionario con los resultados esperados

"""
def medir_metricas(comando):
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

    return {
        "tiempo_total_seg": round(tiempo_total, 4),
        "cpu_user_seg": round(uso_cpu_usuario, 4),
        "cpu_kernel_seg": round(uso_cpu_kernel, 4),
        "memoria_max_kb": round(max_memoria / 1024, 2),
        "codigo_retorno": proceso.returncode,
        "salida": stdout.decode(errors="ignore"),
        "errores": stderr.decode(errors="ignore")
    }
