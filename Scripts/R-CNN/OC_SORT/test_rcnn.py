import os
import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import time
from trackers.ocsort_tracker.ocsort import OCSort

# Secuencias a procesar
SECUENCIAS = [
     "MOT17-06-SDP",
    "MOT17-07-DPM", "MOT17-07-FRCNN", "MOT17-07-SDP",
    "MOT17-08-DPM", "MOT17-08-FRCNN", "MOT17-08-SDP",
    "MOT17-12-DPM", "MOT17-12-FRCNN", "MOT17-12-SDP",
    "MOT17-14-DPM", "MOT17-14-FRCNN", "MOT17-14-SDP"
]

# Paths base
BASE_PATH = "/home/jetson/Documents/MOT17/test"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros
CONF_THRESH = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo de detección (Faster R-CNN)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(DEVICE).eval()

# Tracker OC-SORT
tracker = OCSort(det_thresh=CONF_THRESH)

# Clase "person"
PERSON_CLASS = 1

# Proceso por secuencia
for seq in SECUENCIAS:
    print(f"\n🟢 Procesando secuencia: {seq}")
    IMG_FOLDER = os.path.join(BASE_PATH, seq, "img1")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{seq}.txt")

    image_files = sorted([f for f in os.listdir(IMG_FOLDER) if f.endswith(".jpg")])

    with open(OUTPUT_FILE, "w") as f_out:
        pbar = tqdm(total=len(image_files), desc=f"{seq}")

        for frame_id, img_name in enumerate(image_files):
            img_path = os.path.join(IMG_FOLDER, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"⚠️ Imagen no válida: {img_path}")
                continue

            height, width = image.shape[:2]
            img_tensor = torchvision.transforms.functional.to_tensor(image).to(DEVICE)

            try:
                start = time.time()
                with torch.no_grad():
                    preds = model([img_tensor])[0]
                end = time.time()
            except Exception as e:
                print(f"❌ Error en detección en frame {frame_id + 1}: {e}")
                continue

            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            mask = (labels == PERSON_CLASS) & (scores >= CONF_THRESH)
            person_boxes = boxes[mask]
            person_scores = scores[mask]

            if len(person_boxes) > 0:
                dets = np.hstack((person_boxes, person_scores[:, None]))  # (N, 5)
            else:
                dets = np.empty((0, 5), dtype=np.float32)

            tracks = tracker.update(dets, img_info=(height, width), img_size=(height, width))

            for track in tracks:
                x1, y1, x2, y2, track_id = track
                w = x2 - x1
                h = y2 - y1
                # Salida sin comas, con espacios
                f_out.write(f"{frame_id + 1} {int(track_id)} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} 1 -1 -1 -1\n")

            pbar.set_postfix(time=f"{(end - start):.3f}s")
            pbar.update(1)

    print(f"✅ Finalizado: {OUTPUT_FILE}")
