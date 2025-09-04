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
DETECTIONS_OUTPUT_FILE = os.path.join(os.getcwd(), "detecciones_motchallenge.txt")
YOLO_MODEL = "yolov8n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

# === CABECERA DEL CSV DE RENDIMIENTO ===
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
with open(CSV_RENDIMIENTO, mode='a', newline='') as f_rendimiento, \
     open(DETECTIONS_OUTPUT_FILE, mode='w', newline='') as f_detecciones:

    csv_writer_rendimiento = csv.writer(f_rendimiento)
    csv_writer_detecciones = csv.writer(f_detecciones, delimiter=' ')

    frame_num = 0

    for path in imagenes:
        frame_num += 1
        nombre = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ No se pudo leer {path}")
            continue

        # === DETECCIÓN Y TRACKING ===
        results = model.track(
            img,
            persist=True,
            classes=[0],            # Solo personas
            conf=0.20,               # Umbral de confianza
            iou=0.5,                # IOU para NMS
            tracker=TRACKER_CONFIG,
	    imgsz=1280
        )

        # === MEDIR DESEMPEÑO ===
        tiempo_relativo = time.time() - inicio_global
        #volt, curr, power, ram_used_mb = medir_sistema()
        volt=0
        curr=0
        power=0
        ram_used_mb =0
        # === GUARDAR CSV DE RENDIMIENTO ===
        data_row_rendimiento = [
            nombre,
            round(tiempo_relativo, 3),
            round(ram_used_mb, 2),
            round(volt, 3),
            round(curr, 3),
            round(power, 3)
        ]
        csv_writer_rendimiento.writerow(data_row_rendimiento)

        # === GUARDAR DETECCIONES EN FORMATO MOT ===
        if results and results[0].boxes:
            for box in results[0].boxes:
                bbox = box.xyxy.cpu().numpy().squeeze()
                bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
                bbox_width = bbox_right - bbox_left
                bbox_height = bbox_bottom - bbox_top
                obj_id = int(box.id.cpu().numpy()) if box.id is not None else -1
                conf = round(float(box.conf.cpu().numpy().squeeze()), 3) if box.conf is not None else -1

                row = [
                    frame_num,
                    obj_id,
                    round(bbox_left, 2),
                    round(bbox_top, 2),
                    round(bbox_width, 2),
                    round(bbox_height, 2),
                    conf,
                    -1, -1, -1, -1  # MOT formato extendido
                ]
                csv_writer_detecciones.writerow(row)

        print(f"✅ {nombre} procesada. Tiempo: {tiempo_relativo:.2f}s | RAM: {ram_used_mb} MB | Detecciones: {len(results[0].boxes) if results and results[0].boxes else 0}")

print(f"\n📄 CSV de rendimiento: {CSV_RENDIMIENTO}")
print(f"📄 Detecciones MOTChallenge: {DETECTIONS_OUTPUT_FILE}")
