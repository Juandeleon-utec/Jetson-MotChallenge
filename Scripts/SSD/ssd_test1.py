import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import csv

# === CONFIGURACIÓN ===
IMG_DIR = Path("/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1" )# Ruta a imágenes MOTChallenge
OUT_PATH = Path("detecciones_mobilenet_motchallenge.txt")
CONF_THRESH = 0.2
CLASE_PERSON = 15

# === CARGAR MODELO MobileNet-SSD v2 ===
prototxt = "MobileNetSSD_deploy.prototxt"
weights = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, weights)

# En Jetson Nano: usar CUDA si está disponible
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# === PROCESAR IMÁGENES ===
img_files = sorted(IMG_DIR.glob("*.jpg"))

with open(OUT_PATH, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    for frame_idx, img_path in enumerate(tqdm(img_files, desc="Procesando"), start=1):
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            class_id = int(detections[0, 0, i, 1])

            if conf >= CONF_THRESH and class_id == CLASE_PERSON:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x, y = max(0, x1), max(0, y1)
                w_box = max(0, x2 - x1)
                h_box = max(0, y2 - y1)

                # Formato MOTChallenge: frame, ID(-1), x, y, w, h, conf, -1, -1, -1
                writer.writerow([frame_idx, -1, x, y, w_box, h_box, round(conf, 4), -1, -1, -1])
