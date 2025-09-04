from ultralytics import YOLO
import cv2   
import os
import glob
import motmetrics as mm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


# Cargar el modelo YOLOv8 nano (puede ser yolov8s.pt, etc.)
model = YOLO('yolov8s.pt')

# Ejecutar tracking sobre las imágenes de la secuencia MOT, solo para personas (class_id = 0)
model.track(
    source='/home/jetson/Documents/MOT17/train/MOT17-02-DPM/img1',   # Carpeta de imágenes de la secuencia MOT
    #train/MOT17-02-FRCNN/
    #X:\Carpetas_Personales\MOT17.zip\MOT17\test\MOT17-01-FRCNN\img1\
    tracker='bytetrack.yaml',      # Tracker (puede ser bytetrack o botsort)
    save=True,
    save_txt=True,
    save_conf=True,
    classes=[0],              # SOLO clase 0 = "person"
    project='mot_test',
    name='mot17-04-person',
    show=False
)


# Directorio con resultados YOLOv8 tracking
input_dir = 'mot_test/mot17-04-person/labels'  # ajustá si cambiaste name=
#source='test/MOT17-01-FRCNN/img1',
output_file = 'res/MOT17-01.txt'
video_width = 1920  # Ajustar si la resolución del video es diferente
video_height = 1080

os.makedirs('res', exist_ok=True)

with open(output_file, 'w') as fout:
    for label_file in sorted(glob.glob(os.path.join(input_dir, '*.txt'))):
        frame = int(os.path.basename(label_file).split('.')[0]) + 1
        with open(label_file, 'r') as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue  # Saltear líneas mal formadas

                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                conf = float(parts[5])
                track_id = int(parts[6])

                x = x_center - width / 2
                y = y_center - height / 2

                fout.write(f"{frame}, {track_id}, {x*video_width:.2f}, {y*video_height:.2f}, {width*video_width:.2f}, {height*video_height:.2f}, {conf:.2f}, -1, -1, -1\n")
                
# === ARCHIVOS DE ENTRADA ===
gt_file = "/home/jetson/Documents/MOT17/train/MOT17-02-DPM/gt/gt.txt"
# Ground truth
#source='test/MOT17-01-FRCNN/img1',
res_file = 'res/MOT17-01.txt'                  # Resultados del tracker

# === GENERAR TIMESTAMP PARA ARCHIVOS ===
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = f'resultados_{timestamp}.csv'
png_path = f'resultados_{timestamp}.png'

# === LEER GT ===
gt = pd.read_csv(gt_file, header=None)
gt.columns = ['FrameId', 'Id', 'X', 'Y', 'W', 'H', 'Conf', 'Class', 'Vis']
gt = gt[gt['Class'] == 1]  # Solo personas

# === LEER RESULTADOS DEL TRACKER ===
res = pd.read_csv(res_file, header=None)
res.columns = ['FrameId', 'Id', 'X', 'Y', 'W', 'H', 'Conf', 'Unk1', 'Unk2', 'Unk3']

# === EVALUACIÓN MOT METRICS ===
acc = mm.MOTAccumulator(auto_id=True)

for frame in sorted(gt['FrameId'].unique()):
    gt_frame = gt[gt['FrameId'] == frame]
    res_frame = res[res['FrameId'] == frame]

    gt_boxes = list(zip(gt_frame['X'], gt_frame['Y'], gt_frame['W'], gt_frame['H']))
    res_boxes = list(zip(res_frame['X'], res_frame['Y'], res_frame['W'], res_frame['H']))

    dist = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)

    acc.update(gt_frame['Id'].tolist(), res_frame['Id'].tolist(), dist)

# === GENERAR MÉTRICAS Y GUARDAR CSV ===
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='MOT17-01')
summary_rounded = summary.round(3)
summary_rounded.to_csv(csv_path)
print(f"✅ CSV guardado: {csv_path}")

# === GRAFICAR MÉTRICAS CLAVE ===
metricas_clave = ['idf1', 'idp', 'idr', 'recall', 'precision', 'mota', 'motp']
metricas = summary_rounded.loc['MOT17-01'].reindex(metricas_clave) * 100

plt.figure(figsize=(10, 6))
bars = plt.bar(metricas.index.str.upper(), metricas.values, color='skyblue')
plt.ylabel('Porcentaje (%)')
plt.ylim(0, 100)
plt.title(f'Métricas de Evaluación - MOT17-01 ({timestamp})')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(png_path)
plt.show()
print(f"✅ Gráfico guardado: {png_path}")