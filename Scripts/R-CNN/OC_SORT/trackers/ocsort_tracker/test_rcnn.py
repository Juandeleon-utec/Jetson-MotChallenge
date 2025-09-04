import os
import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
#import sys
#sys.path.append('/home/jetson/Documents/mot/R-CNN/OC_SORT/trackers/ocsort_tracker')
from ocsort import OCSort  # Asegurate de tener OC-SORT en tu PYTHONPATH
from torchvision.transforms import functional as F

# Configuración
SEQUENCE_PATH = "/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1"
OUTPUT_TXT = "output/MOT17-02-DPM.txt"
CONF_THRESH = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar detector Faster R-CNN preentrenado
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(DEVICE)

# Inicializar tracker OC-SORT
tracker = OCSort()

# Obtener lista de imágenes
image_files = sorted([f for f in os.listdir(SEQUENCE_PATH) if f.endswith('.jpg')])

# Crear carpeta de salida
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

# Procesar secuencia
with open(OUTPUT_TXT, 'w') as f_out:
    for frame_id, img_name in enumerate(tqdm(image_files, desc="Procesando")):
        img_path = os.path.join(SEQUENCE_PATH, img_name)
        img = cv2.imread(img_path)
        img_tensor = F.to_tensor(img).to(DEVICE)

        with torch.no_grad():
            detections = model([img_tensor])[0]

        # Filtrar solo clase "person" (COCO class 1)
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()

        person_mask = (labels == 1) & (scores >= CONF_THRESH)
        boxes = boxes[person_mask]
        scores = scores[person_mask]

        # Formato para OC-SORT: [x1, y1, x2, y2, score]
        dets = np.concatenate([boxes, scores[:, None]], axis=1)

        tracks = tracker.update(dets)

        for d in tracks:
            x1, y1, x2, y2, track_id = d
            w = x2 - x1
            h = y2 - y1
            f_out.write(f"{frame_id+1},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
