import time
import psutil
import subprocess
import json
import os
from datetime import datetime

"""
calc_metrics(command, slam_name=None): measures process resource usage.
If slam_name is specified, it saves the results as metrics.json
under results/<slam_name>/<timestamp>/
"""

def calc_metrics(command, slam_name=None):
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps_process = psutil.Process(process.pid)

    cpu_user = 0
    cpu_kernel = 0
    max_memory = 0

    try:
        while process.poll() is None:
            cpu_times = ps_process.cpu_times()
            cpu_user = cpu_times.user
            cpu_kernel = cpu_times.system
            mem_info = ps_process.memory_info()
            max_memory = max(max_memory, mem_info.rss)
            time.sleep(0.1)

        stdout, stderr = process.communicate()
    except psutil.NoSuchProcess:
        stdout, stderr = b"", b"Process not available"

    total_time = time.time() - start_time

    results = {
        "total_time_sec": round(total_time, 3),
        "cpu_user_sec": round(cpu_user, 4),
        "cpu_kernel_sec": round(cpu_kernel, 4),
        "max_memory_kb": round(max_memory / 1024, 2),
        "return_code": process.returncode,
        "stdout": stdout.decode(errors="ignore"),
        "stderr": stderr.decode(errors="ignore")
    }

    if slam_name:
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        output_dir = os.path.join("results", slam_name, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "metrics.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    return results
