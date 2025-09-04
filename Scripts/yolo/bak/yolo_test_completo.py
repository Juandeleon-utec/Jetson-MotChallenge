import torch
import os
import time
import csv
import cv2
from ultralytics import YOLO
from jtop import jtop
from glob import glob

# === CONFIGURACIÓN GENERAL ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGES_DIR = "/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1"
CSV_RENDIMIENTO = os.path.join(os.getcwd(), "rendimiento_tiempo_relativo.csv")
YOLO_MODEL = "yolov8n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

# === CABECERA DEL CSV ===
headers = ["imagen", "tiempo_relativo_s", "ram_used_MB", "volt", "curr", "power"]
if not os.path.exists(CSV_RENDIMIENTO):
    with open(CSV_RENDIMIENTO, mode='w', newline='') as f:
        csv.writer(f).writerow(headers)

# === MEDICIÓN DEL SISTEMA ===
def medir_sistema():
    volt, curr, power, ram_used_mb = 0.0, 0.0, 0.0, 0.0
    with jtop() as jetson:
        if jetson.ok():
            power_data = jetson.power.get('tot', {})
            volt = power_data.get('volt', 0.0)
            curr = power_data.get('curr', 0.0)
            power = power_data.get('power', 0.0)
            ram_used_mb = jetson.memory['RAM']['used']
    return volt, curr, power, ram_used_mb

# === CARGAR MODELO YOLO ===
model = YOLO(YOLO_MODEL).to(DEVICE)
model.fuse()
print(f"🧠 Modelo cargado en {DEVICE}")

# === LISTA DE IMÁGENES ===
imagenes = sorted(glob(os.path.join(IMAGES_DIR, "*.jpg")))

# === INICIO DE TIEMPO GLOBAL ===
inicio_global = time.time()

# === PROCESAMIENTO POR IMAGEN ===
# Open the CSV file in append mode outside the loop
with open(CSV_RENDIMIENTO, mode='a', newline='') as f:
    csv_writer = csv.writer(f)

    for path in imagenes:
        nombre = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ No se pudo leer {path}")
            continue

        # === DETECCIÓN Y TRACKING ===
        _ = model.track(img, persist=True, classes=[0], conf=0.2, tracker=TRACKER_CONFIG)

        # === MEDIR DESEMPEÑO ===
        tiempo_relativo = time.time() - inicio_global
        volt, curr, power, ram_used_mb = medir_sistema()

        # === GUARDAR MEDICIÓN ===
        # Write the data for the current image directly to the CSV
        data_row = [
            nombre,
            round(tiempo_relativo, 3),
            round(ram_used_mb, 2),
            round(volt, 3),
            round(curr, 3),
            round(power, 3)
        ]
        csv_writer.writerow(data_row)

        print(f"✅ {nombre} procesada. Tiempo: {tiempo_relativo:.2f}s | RAM: {ram_used_mb} MB")

print(f"\n📄 CSV guardado: {CSV_RENDIMIENTO}")
print(f"📷 Total de imágenes procesadas: {len(imagenes)}") # Use len(imagenes) here as we are writing row by row
