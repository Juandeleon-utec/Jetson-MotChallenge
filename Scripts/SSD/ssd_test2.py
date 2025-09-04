import cv2
import os
import numpy as np
from sort import Sort

# === Configuracin ===
BASE_DIR = "/home/jetson/Documents/MOT17/train"
# === SECUENCIAS A PROCESAR ===
SECUENCIAS = [
    "MOT17-02-DPM", "MOT17-02-FRCNN", "MOT17-02-SDP",
    "MOT17-04-DPM", "MOT17-04-FRCNN", "MOT17-04-SDP",
    "MOT17-05-DPM", "MOT17-05-FRCNN", "MOT17-05-SDP",
    "MOT17-09-DPM", "MOT17-09-FRCNN", "MOT17-09-SDP",
    "MOT17-10-DPM", "MOT17-10-FRCNN", "MOT17-10-SDP",
    "MOT17-11-DPM", "MOT17-11-FRCNN", "MOT17-11-SDP",
    "MOT17-13-DPM", "MOT17-13-FRCNN", "MOT17-13-SDP"
]

PROTOTXT = "MobileNetSSD_deploy.prototxt"
WEIGHTS = "MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.2

# Clases MobileNet-SSD (solo "person")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
PERSON_CLASS_ID = CLASSES.index("person")

net = cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

for secuencia in SECUENCIAS:
    print(f"\n?? Procesando secuencia: {secuencia}")
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
                print(f"?? No se pudo leer {frame_path}, saltando.")
                frame_id += 1
                continue

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            dets = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                if confidence > CONFIDENCE_THRESHOLD and class_id == PERSON_CLASS_ID:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    dets.append([x1, y1, x2, y2, confidence])

            if len(dets) > 0:
                dets_np = np.array(dets)
            else:
                dets_np = np.empty((0, 5))

            tracks = tracker.update(dets_np)

            for track in tracks:
                x1, y1, x2, y2, track_id = track
                width = x2 - x1
                height = y2 - y1
                f_out.write(f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},1,-1,-1,-1\n")

            frame_id += 1

print("\n? Procesamiento completo, archivos generados por secuencia.")
