import time
import torch
import os
import csv
import cv2
from ultralytics import YOLO
from glob import glob
#from jtop import jtop  # Descomenta si usás medición de sistema

# === CONFIGURACIÓN GENERAL ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/jetson/Documents/MOT17/test"
YOLO_MODEL = "best_utec.pt"
TRACKER_CONFIG = "bytetrack.yaml"
CSV_RENDIMIENTO = os.path.join(os.getcwd(), "rendimiento_tiempo_relativo.csv")

# === SECUENCIAS A PROCESAR ===
SECUENCIAS = [
    "MOT17-01-DPM", "MOT17-01-FRCNN", "MOT17-01-SDP",
    "MOT17-03-DPM", "MOT17-03-FRCNN", "MOT17-03-SDP",
    "MOT17-06-DPM", "MOT17-06-FRCNN", "MOT17-06-SDP",
    "MOT17-07-DPM", "MOT17-07-FRCNN", "MOT17-07-SDP",
    "MOT17-08-DPM", "MOT17-08-FRCNN", "MOT17-08-SDP",
    "MOT17-12-DPM", "MOT17-12-FRCNN", "MOT17-12-SDP",
    "MOT17-14-DPM", "MOT17-14-FRCNN", "MOT17-14-SDP"
]



# === CABECERA CSV (opcional) ===
headers = ["secuencia", "imagen", "tiempo_relativo_s"]
if not os.path.exists(CSV_RENDIMIENTO):
    with open(CSV_RENDIMIENTO, mode='w', newline='') as f:
        csv.writer(f).writerow(headers)

# === CARGAR MODELO UNA SOLA VEZ ===
model = YOLO(YOLO_MODEL).to(DEVICE)
model.fuse()
print(f"🧠 Modelo cargado en {DEVICE}")

inicio_global = time.time()

# === PROCESAR TODAS LAS SECUENCIAS ===
for secuencia in SECUENCIAS:
    print(f"\n📂 Procesando secuencia: {secuencia}")
    model = YOLO(YOLO_MODEL).to(DEVICE)
    model.fuse()
    IMAGES_DIR = os.path.join(BASE_DIR, secuencia, "img1")
    DETECTIONS_OUTPUT_FILE = os.path.join(os.getcwd(), secuencia + ".txt")

    imagenes = sorted(glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not imagenes:
        print(f"⚠️  No se encontraron imágenes en {IMAGES_DIR}")
        continue

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

            results = model.track(
                img,
                persist=True,
                classes=[0],       # Solo personas
                conf=0.20,
                iou=0.5,
                tracker=TRACKER_CONFIG,
                imgsz=1280
            )

            tiempo_relativo = time.time() - inicio_global

            # Guardar rendimiento
            csv_writer_rendimiento.writerow([
                secuencia,
                nombre,
                round(tiempo_relativo, 3)
            ])

            # Guardar detecciones en formato MOT
            if results and results[0].boxes:
                seen_ids = set()
                for box in results[0].boxes:
                    if box.id is None:
                        continue
                    obj_id = int(box.id.cpu().numpy())
                    if (frame_num, obj_id) in seen_ids:
                        continue
                    seen_ids.add((frame_num, obj_id))

                    bbox = box.xyxy.cpu().numpy().squeeze()
                    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
                    bbox_width = bbox_right - bbox_left
                    bbox_height = bbox_bottom - bbox_top
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

            print(f"✅ {secuencia}/{nombre} procesada. Tiempo: {tiempo_relativo:.2f}s")

print("\n🏁 Procesamiento finalizado para todas las secuencias.")
