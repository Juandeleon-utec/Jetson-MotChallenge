# -*- coding: utf-8 -*-
import torch
import time
import csv
import os
from datetime import datetime
from glob import glob
from ultralytics import YOLO
import cv2
from jtop import jtop

# === CONFIGURACIÓN ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, '/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1')  # Carpeta con imágenes MOT
CSV_METRICS = os.path.join(SCRIPT_DIR, "personas_rendimiento.csv")
CSV_DETECCIONES = os.path.join(SCRIPT_DIR, "personas_detecciones.csv")

# === MODELO ===
model = YOLO("yolov8n.pt")
model.to(DEVICE)
model.fuse()

# === HEADERS ===
headers_metricas = [
    "timestamp", "imagen", "duracion_s", "gpu_mem_used_MB",
    "mem_reserved_MB", "Volt", "Curr", "power"
]
headers_detecciones = [
    "timestamp", "imagen", "clase", "confianza", "x1", "y1", "x2", "y2"
]
print(f"PyTorch usará: {DEVICE}")


# Crear CSVs si no existen
for path, headers in [(CSV_METRICS, headers_metricas), (CSV_DETECCIONES, headers_detecciones)]:
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as f:
            csv.writer(f).writerow(headers)

def medir_sistema():
    volt, curr, power, gpu_mem_used_mb = 0.0, 0.0, 0.0, 0.0
    with jtop() as jetson:
        if jetson.ok():
            power_data = jetson.power.get('tot', {})
            volt = power_data.get('volt', 0.0)
            curr = power_data.get('curr', 0.0)
            power = power_data.get('power', 0.0)
            gpu_mem_used_mb = jetson.gpu.get('mem', {}).get('used', 0.0)
    return volt, curr, power, gpu_mem_used_mb

# === PROCESAR IMÁGENES ===
image_files = sorted(glob(os.path.join(IMAGES_DIR, "*.jpg")))
print(f"Se encontraron {len(image_files)} imágenes en {IMAGES_DIR}")

for img_path in image_files:
    img_name = os.path.basename(img_path)
    print(f"\nProcesando: {img_name}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error al leer {img_path}")
        continue

    start_time = time.time()
    results = model(frame, verbose=False, classes=[0])[0]  # Solo personas
    end_time = time.time()
    duracion = end_time - start_time

    # Obtener detecciones de personas
    detecciones = []
    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detecciones.append([
            timestamp, img_name, "persona", round(conf, 2), x1, y1, x2, y2
        ])

    # Métricas sistema
    volt, curr, power, gpu_mem_used_mb = medir_sistema()
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**2

    # Guardar métricas
    with open(CSV_METRICS, mode='a', newline='') as f:
        csv.writer(f).writerow([
            timestamp, img_name, round(duracion, 3),
            round(gpu_mem_used_mb, 2),
            round(mem_reserved, 2), volt, curr, power
        ])

    # Guardar detecciones
    with open(CSV_DETECCIONES, mode='a', newline='') as f:
        csv.writer(f).writerows(detecciones)

    print(f"{img_name} procesada. {len(detecciones)} personas detectadas.")

print("\nResultado de métricas:", CSV_METRICS)
print("Resultado de detecciones:", CSV_DETECCIONES)
