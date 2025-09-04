import cv2
import numpy as np
import os
from sort import Sort

# === Configuración ===
BASE_DIR = "/home/jetson/Documents/MOT17/train"
SECUENCIAS = [
    "MOT17-02-DPM", "MOT17-02-FRCNN", "MOT17-02-SDP",
    "MOT17-04-DPM", "MOT17-04-FRCNN", "MOT17-04-SDP",
    "MOT17-05-DPM", "MOT17-05-FRCNN", "MOT17-05-SDP",
    "MOT17-09-DPM", "MOT17-09-FRCNN", "MOT17-09-SDP",
    "MOT17-10-DPM", "MOT17-10-FRCNN", "MOT17-10-SDP",
    "MOT17-11-DPM", "MOT17-11-FRCNN", "MOT17-11-SDP",
    "MOT17-13-DPM", "MOT17-13-FRCNN", "MOT17-13-SDP"
]

YOLO_ONNX = "best.onnx"  # tu modelo YOLOv8
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0  # En YOLOv8, 'person' suele ser clase 0

# Cargar YOLOv8 ONNX con OpenCV DNN
net = cv2.dnn.readNetFromONNX(YOLO_ONNX)

for secuencia in SECUENCIAS:
    print(f"\n🔄 Procesando secuencia: {secuencia}")
    img_dir = os.path.join(BASE_DIR, secuencia, "img1")
    output_txt = f"{secuencia}.txt"
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    frame_id = 1
    with open(output_txt, "w") as f_out:
        for img_file in image_files:
            frame_path = os.path.join(img_dir, img_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"⚠ No se pudo leer {frame_path}, saltando.")
                frame_id += 1
                continue

            (h, w) = frame.shape[:2]
            # Preparar blob para YOLOv8
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()

            # YOLOv8 en OpenCV DNN produce shape [1, N, 84] -> [cx, cy, w, h, conf, cls1, cls2, ...]
            detections = outputs[0]
            dets = []

            for det in detections:
                confidence = det[4]
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                class_id = np.argmax(det[5:])
                if class_id != PERSON_CLASS_ID:
                    continue
                scores = det[5:]
                score = scores[class_id] * confidence
                if score < CONFIDENCE_THRESHOLD:
                    continue
                # Convertir de cx,cy,w,h a x1,y1,x2,y2
                cx, cy, bw, bh = det[0:4]
                x1 = (cx - bw / 2) * w / 640
                y1 = (cy - bh / 2) * h / 640
                x2 = (cx + bw / 2) * w / 640
                y2 = (cy + bh / 2) * h / 640
                dets.append([x1, y1, x2, y2, float(score)])

            dets_np = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
            tracks = tracker.update(dets_np)

            for track in tracks:
                x1, y1, x2, y2, track_id = track
                width = x2 - x1
                height = y2 - y1
                f_out.write(f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},1,-1,-1,-1\n")

            frame_id += 1

print("\n✅ Procesamiento completo, archivos generados por secuencia.")

