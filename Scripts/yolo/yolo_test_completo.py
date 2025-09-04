
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

# === ARCHIVO DE SALIDA PARA DETECCIONES MOTCHALLENGE ===
# Define the output file for detections. You might want to make this dynamic
# based on the sequence being processed if you handle multiple sequences.
DETECTIONS_OUTPUT_FILE = os.path.join(os.getcwd(), "detecciones_motchallenge.txt")

# === CABECERA DEL CSV ===
headers = ["imagen", "tiempo_relativo_s", "ram_used_MB", "volt", "curr", "power"]
if not os.path.exists(CSV_RENDIMIENTO):
    with open(CSV_RENDIMIENTO, mode='w', newline='') as f:
        csv.writer(f).writerow(headers)

# === CABECERA DEL ARCHIVO DE DETECCIONES (OPCIONAL, DEPENDIENDO DEL FORMATO) ===
# Algunas implementaciones de MOTChallenge no tienen cabecera en el archivo de detecciones.
# Si necesitas una, descomenta y ajusta la siguiente sección.
# detections_headers = "frame id bbox_left bbox_top bbox_width bbox_height conf l r t i"
# if not os.path.exists(DETECTIONS_OUTPUT_FILE):
#     with open(DETECTIONS_OUTPUT_FILE, mode='w') as f:
#         f.write(detections_headers + "\n")


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

# === INICIO DE TIPO GLOBAL ===
inicio_global = time.time()

# === PROCESAMIENTO POR IMAGEN ===
# Open the CSV file in append mode outside the loop
with open(CSV_RENDIMIENTO, mode='a', newline='') as f_rendimiento, \
     open(DETECTIONS_OUTPUT_FILE, mode='a', newline='') as f_detecciones: # Open detections file too

    csv_writer_rendimiento = csv.writer(f_rendimiento)
    csv_writer_detecciones = csv.writer(f_detecciones, delimiter=' ') # MOTChallenge often uses space delimiter

    # Initialize frame counter for MOTChallenge format
    frame_num = 0

    for path in imagenes:
        frame_num += 1 # Increment frame number for each image
        nombre = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ No se pudo leer {path}")
            continue

        # === DETECCIÓN Y TRACKING ===
        # Capture the results of the track method
        results = model.track(img, persist=True, classes=[0], conf=0.2, tracker=TRACKER_CONFIG)

        # === MEDIR DESEMPEÑO ===
        tiempo_relativo = time.time() - inicio_global
        volt, curr, power, ram_used_mb = medir_sistema()

        # === GUARDAR MEDICIÓN DE RENDIMIENTO ===
        data_row_rendimiento = [
            nombre,
            round(tiempo_relativo, 3),
            round(ram_used_mb, 2),
            round(volt, 3),
            round(curr, 3),
            round(power, 3)
        ]
        csv_writer_rendimiento.writerow(data_row_rendimiento)

        # === GUARDAR DETECCIONES EN FORMATO MOTCHALLENGE ===
        # Process the results to extract detection data
        if results and results.boxes: # Check if there are results and boxes
            for box in results.boxes:
                # Extract box coordinates and other info
                # MOTChallenge format: frame id bbox_left bbox_top bbox_width bbox_height conf l r t i
                # Note: conf, l, r, t, i are often placeholders or specific to certain evaluations.
                # Adjust extraction and formatting based on your specific needs.
                bbox_left, bbox_top, bbox_right, bbox_bottom = box.xyxy.cpu().numpy()
                bbox_width = bbox_right - bbox_left
                bbox_height = bbox_bottom - bbox_top
                obj_id = int(box.id.cpu().numpy()) if box.id is not None else -1 # Get object ID
                conf = round(box.conf.cpu().numpy(), 3) if box.conf is not None else -1 # Get confidence

                # Format the detection data for MOTChallenge
                detection_data_row = [
                    frame_num,
                    obj_id,
                    round(bbox_left, 2),
                    round(bbox_top, 2),
                    round(bbox_width, 2),
                    round(bbox_height, 2),
                    conf,
                    -1, # Placeholder for 'l'
                    -1, # Placeholder for 'r'
                    -1, # Placeholder for 't'
                    -1  # Placeholder for 'i'
                ]
                csv_writer_detecciones.writerow(detection_data_row)

        print(f"✅ {nombre} procesada. Tiempo: {tiempo_relativo:}.")
        
