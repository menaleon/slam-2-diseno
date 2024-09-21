import cv2
import os

def create_video_from_images_left(image_folder, output_video_file, fps=30):
    # Obtener la lista de imágenes en el folder y ordenarlas
    images = [img for img in os.listdir(image_folder) if img.endswith("_left.png")]
    images.sort(key=lambda x: int(x.split('_')[0]))  # Ordenar por el número antes de '_left'

    # Leer la primera imagen para obtener las dimensiones
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Definir el codec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para video MP4
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Escribir cada imagen al video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Liberar el objeto VideoWriter
    video.release()
    print(f"Video guardado como {output_video_file}")

def create_video_from_images(image_folder, output_video_file, fps=30):
    # Obtener la lista de imágenes en el folder y ordenarlas
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Ordenar por el valor flotante del nombre del archivo (timestamp)
    images.sort(key=lambda x: float(os.path.splitext(x)[0]))  # Convertir el nombre antes de la extensión a float

    # Comprobar si se encontraron imágenes
    if len(images) == 0:
        raise ValueError("No se encontraron imágenes en el formato esperado.")

    # Leer la primera imagen para obtener las dimensiones
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Definir el codec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para video MP4
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Escribir cada imagen al video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Liberar el objeto VideoWriter
    video.release()
    print(f"Video guardado como {output_video_file}")

# Ruta a la carpeta de imágenes
image_folder = '/home/faleivac/Documents/Universidad/2024/slam/datasets/sfm_house_loop_mono/sfm_house_loop/rgb/'
output_video_file = 'output_video2.mp4'  # Puedes cambiar el nombre o la ruta del archivo de salida
fps = 30  # Cambia el valor de fps si deseas otro frame rate

# Crear el video
create_video_from_images(image_folder, output_video_file, fps)


"""
# Ruta a la carpeta de imágenes
image_folder = '/home/faleivac/Documents/Universidad/2024/slam/datasets/office2_sample_P003/P003/image_left/'
output_video_file = 'output_video.mp4'  # Puedes cambiar el nombre o la ruta del archivo de salida
fps = 30  # Cambia el valor de fps si deseas otro frame rate

# Crear el video
create_video_from_images_left(image_folder, output_video_file, fps)
"""

