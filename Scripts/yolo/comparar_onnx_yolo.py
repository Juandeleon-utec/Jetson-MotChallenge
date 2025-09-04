import time
import cv2
import torch
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

# === CONFIGURACIÓN ===
IMG_DIR = Path("/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1")  # Ruta al directorio de imágenes del MOTChallenge
CLASE_INTERES = 0  # "person" en COCO
CONF_THRES = 0.25
IOU_THRES = 0.45
MAX_FRAMES = 100

# === MODELO PYTORCH ===
model_pt = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_pt.to(device)

# === MODELO ONNX ===
ort_session = ort.InferenceSession("yolov8n.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# === PREPROCESAMIENTO ===
def preprocess(image, size=640):
    img = cv2.resize(image, (size, size))
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR->RGB, CHW, normalize
    img = np.expand_dims(img, axis=0)
    return img

# === INFERENCIA ONNX ===
def infer_onnx(image):
    img = preprocess(image)
    outputs = ort_session.run(None, {"images": img})[0]
    return outputs

# === COMPARACIÓN ===
def comparar_inferencia():
    img_files = sorted(IMG_DIR.glob("*.jpg"))[:MAX_FRAMES]

    resultados = {
        "PyTorch": {"fps": 0, "tiempo": 0, "detecciones": 0},
        "ONNX": {"fps": 0, "tiempo": 0, "detecciones": 0}
    }

    # === INFERENCIA CON PYTORCH ===
    start = time.time()
    for img_path in img_files:
        frame = cv2.imread(str(img_path))
        results = model_pt(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
        detections = results.boxes.cls.cpu().numpy()
        resultados["PyTorch"]["detecciones"] += np.sum(detections == CLASE_INTERES)
    end = time.time()
    resultados["PyTorch"]["fps"] = len(img_files) / (end - start)
    resultados["PyTorch"]["tiempo"] = (end - start) / len(img_files)

    # === INFERENCIA CON ONNX ===
    start = time.time()
    for img_path in img_files:
        frame = cv2.imread(str(img_path))
        outputs = infer_onnx(frame)[0]
        detections = outputs[outputs[:, 4] > CONF_THRES]
        detections = detections[detections[:, 5] == CLASE_INTERES]
        resultados["ONNX"]["detecciones"] += len(detections)
    end = time.time()
    resultados["ONNX"]["fps"] = len(img_files) / (end - start)
    resultados["ONNX"]["tiempo"] = (end - start) / len(img_files)

    return resultados

# === EJECUCIÓN ===
res = comparar_inferencia()
print("\n--- COMPARACIÓN DE RENDIMIENTO ---")
for k, v in res.items():
    print(f"\n>> {k}")
    print(f"FPS promedio         : {v['fps']:.2f}")
    print(f"Tiempo por imagen    : {v['tiempo']:.4f} s")
    print(f"Detecciones 'persona': {v['detecciones']}")
