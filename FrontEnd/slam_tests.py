# testing.py
import time
import psutil
import json
import os

class Testing:

    def __init__(self, path, algorithm_type):
        self.cpu_usage = []
        self.memory_usage = []
        self.start_time = None
        self.end_time = None
        self.iteration = 0
        self.path = path
        self.algorithm_type = algorithm_type

    def take_measure(self):
        if self.iteration == 0:
            self.start_time = time.time()
        self.cpu_usage.append(psutil.cpu_percent(interval=None))
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.iteration += 1

    def end_testing(self):
        self.end_time = time.time()

    def get_summary(self):
        if not self.cpu_usage or not self.memory_usage:
            return None
        
        average_cpu_usage = sum(self.cpu_usage) / len(self.cpu_usage)
        average_memory_usage = sum(self.memory_usage) / len(self.memory_usage)
        execution_time = self.end_time - self.start_time if self.end_time else 0

        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "average_cpu_usage": average_cpu_usage,
            "average_memory_usage": average_memory_usage,
            "execution_time": execution_time
        }

    def save_data(self):
        data = self.get_summary()
        if data is None:
            print("No hay datos disponibles para guardar.")
            return
        
        file_path = "results.json"
        try:
            # Intentar cargar el contenido del archivo JSON si existe y no está vacío
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, "r") as file:
                    results = json.load(file)
            else:
                results = {}
        except json.JSONDecodeError:
            # Si hay un error al cargar el JSON, se crea un diccionario vacío
            results = {}

        # Crear el key basado en el path y el tipo de algoritmo
        key = f"{self.path}\\{self.algorithm_type}"

        # Actualizar o agregar la entrada para el key específico
        results[key] = data

        # Guardar los datos actualizados en el archivo JSON
        with open(file_path, "w") as file:
            json.dump(results, file, indent=4)

        print(f"Datos guardados en {file_path} para el key: {key}")


import json
import os

def normalize_json_values(file_path):
    if not os.path.exists(file_path):
        print("El archivo JSON no existe.")
        return
    
    with open(file_path, "r") as file:
        data = json.load(file)

    min_length = min(len(entry["cpu_usage"]) for entry in data.values())
    
    def reduce_list(values, target_length):
        """ Reduce la lista a la longitud objetivo usando promedios """
        chunk_size = len(values) / target_length
        reduced_list = []
        for i in range(target_length):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            chunk = values[start:end]
            reduced_list.append(sum(chunk) / len(chunk))
        return reduced_list

    for key, entry in data.items():
        entry["cpu_usage"] = reduce_list(entry["cpu_usage"], min_length)
        entry["memory_usage"] = reduce_list(entry["memory_usage"], min_length)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"JSON normalizado y guardado en {file_path}")

normalize_json_values("results.json")
