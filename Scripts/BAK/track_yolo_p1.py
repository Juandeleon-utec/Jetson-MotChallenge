from ultralytics import YOLO
import os

# Cargar modelo YOLOv8s
model = YOLO('yolov8s.pt')

# Crear carpeta de resultados
os.makedirs('mot_test', exist_ok=True)

# Ejecutar detección y tracking
model.track(
    source='/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1',
    tracker='bytetrack.yaml',
    save=False,              # No guardar imágenes procesadas para ahorrar espacio/RAM
    save_txt=True,           # Guardar resultados como YOLO TXT (por frame)
    save_conf=False,         # Desactivado para reducir tamaño
    classes=[0],             # Solo personas
    project='mot_test',
    name='mot17-02',
    show=False,
    imgsz=640                # Reduce uso de RAM/GPU
)

print("✅ Etapa 1 completada: resultados en 'mot_test/mot17-02/labels'")
