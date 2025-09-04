import os
import time
import csv
import torch
import torchvision
from glob import glob
from datetime import datetime
from PIL import Image
from torchvision.transforms import functional as F
from jtop import jtop

# === CONFIGURACIÓN ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, '/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1')
CSV_METRICS = os.path.join(SCRIPT_DIR, "fasterrcnn_rendimiento.csv")
CSV_DETECCIONES = os.path.join(SCRIPT_DIR, "fasterrcnn_detecciones.csv")

CONF_THRESHOLD = 0.5

# === MODELO ===
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(DEVICE).eval()

# === HEADERS ===
headers_metricas = [
    "timestamp", "imagen", "duracion_s", "gpu_mem_used_MB",
    "mem_reserved_MB", "Volt", "Curr", "power"
]
headers_detecciones = [
    "timestamp", "imagen", "clase", "confianza", "x1", "y1", "x2", "y2"
]

# Crear CSVs si no existen
for path, headers in [(CSV_METRICS, headers_metricas), (CSV_DETECCIONES, headers_detecciones)]:
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as f:
            csv.writer(f).writerow(headers)

def medir_sistema():
    volt, curr, power, gpu_mem_used_mb = 0.0, 0.0, 0.0, 0.0
    with jtop() as jetson:
        if jetson.ok():
            power_data = jetson.power.get('tot', {})
            volt = power_data.get('volt', 0.0)
            curr = power_data.get('curr', 0.0)
            power = power_data.get('power', 0.0)
            gpu_mem_used_mb = jetson.gpu.get('mem', {}).get('used', 0.0)
    return volt, curr, power, gpu_mem_used_mb

# === PROCESAR IMÁGENES ===
image_files = sorted(glob(os.path.join(IMAGES_DIR, "*.jpg")))
print(f"📷 Se encontraron {len(image_files)} imágenes en: {IMAGES_DIR}")
print(f"🖥️ Dispositivo en uso: {DEVICE}")

for img_path in image_files:
    img_name = os.path.basename(img_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🖼 Procesando: {img_name}")

    try:
        image = Image.open(img_path).convert("RGB")
    except:
        print(f"⚠️ Error al abrir {img_path}")
        continue

    tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        output = model(tensor)[0]
    end_time = time.time()
    duracion = end_time - start_time

    # === Filtrar detecciones ===
    detecciones = []
    for idx in range(len(output["boxes"])):
        score = float(output["scores"][idx])
        label = int(output["labels"][idx])
        if label == 1 and score >= CONF_THRESHOLD:  # Persona = clase 1 en COCO
            box = output["boxes"][idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            detecciones.append([
                timestamp, img_name, "persona", round(score, 2), x1, y1, x2, y2
            ])

    # === Métricas sistema ===
    volt, curr, power, gpu_mem_used_mb = medir_sistema()
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**2 if DEVICE.type == "cuda" else 0.0

    with open(CSV_METRICS, mode='a', newline='') as f:
        csv.writer(f).writerow([
            timestamp, img_name, round(duracion, 3),
            round(gpu_mem_used_mb, 2), round(mem_reserved, 2),
            volt, curr, power
        ])

    with open(CSV_DETECCIONES, mode='a', newline='') as f:
        csv.writer(f).writerows(detecciones)

    print(f"✅ {img_name}: {len(detecciones)} personas detectadas en {duracion:.2f}s")

print("\n📁 CSV de métricas:", CSV_METRICS)
print("📁 CSV de detecciones:", CSV_DETECCIONES)
