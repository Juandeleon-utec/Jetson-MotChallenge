import cv2
import numpy as np
import os
import time
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

YOLO_ONNX = "best.onnx"  # tu modelo YOLOv8 exportado
CONFIDENCE_THRESHOLD = 0.2
PERSON_CLASS_ID = 0  # persona en YOLOv8 normalmente es clase 0

# Cargar red YOLOv8 ONNX
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

            start_time = time.time()

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()

            detections = outputs[0]
            dets = []

            for det in detections:
                confidence = det[4]
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                class_id = np.argmax(det[5:])
                if class_id != PERSON_CLASS_ID:
                    continue
                score = det[5+class_id] * confidence
                if score < CONFIDENCE_THRESHOLD:
                    continue

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

            tiempo_relativo = time.time() - start_time
            print(f"✅ {img_file} procesada. Tiempo: {tiempo_relativo:.2f}s | Detecciones: {len(dets)}")

            # # Para mostrar ventana (opcional, puede reducir rendimiento)
            # for track in tracks:
            #     x1, y1, x2, y2, track_id = track
            #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            #     cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1)-10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imshow("Tracking", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            frame_id += 1

print("\n✅ Procesamiento completo, archivos generados por secuencia.")

