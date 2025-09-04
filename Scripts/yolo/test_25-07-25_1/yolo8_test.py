import torch
import os
from ultralytics import YOLO
from datetime import datetime

# === CONFIGURACIÓN ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta directa a la secuencia de imágenes MOT
IMAGES_DIR = "/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1"

# Configuración de salida del seguimiento
TRACKER_CONFIG = "bytetrack.yaml"  # Puedes cambiar por "botsort.yaml" si lo deseas
PROYECTO = "seguimiento_yolo"
NOMBRE_SECUENCIA = "MOT17-02-person"

# === CARGAR MODELO ===
model = YOLO("yolov8n.pt").to(DEVICE)
model.fuse()
print(f"[{datetime.now()}] PyTorch usará: {DEVICE}")

# === SEGUIMIENTO (detección + tracking) ===
model.track(
    source=IMAGES_DIR,            # Carpeta con las imágenes
    tracker=TRACKER_CONFIG,       # Configuración del tracker
    save=True,                    # Guarda el video y resultados
    save_txt=True,                # Exporta archivos .txt (formato MOTChallenge)
    save_conf=True,               # Guarda confianza en los .txt
    classes=[0],                  # Solo personas
    project=PROYECTO,             # Carpeta raíz para resultados
    name=NOMBRE_SECUENCIA,       # Subcarpeta para esta secuencia
    show=False                    # No mostrar en pantalla
)

# Ruta final de resultados
output_dir = os.path.join("runs", "track", PROYECTO, NOMBRE_SECUENCIA)
print(f"\nSeguimiento completado.")
print(f"Resultados guardados en: {output_dir}")
print(f"Archivos MOTChallenge en: {os.path.join(output_dir, 'labels')}")
