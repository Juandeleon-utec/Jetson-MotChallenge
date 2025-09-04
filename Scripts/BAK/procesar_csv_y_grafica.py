import os
import glob
import pandas as pd
import motmetrics as mm
import matplotlib.pyplot as plt
from datetime import datetime

# === Parámetros ===
input_dir = 'mot_test/mot17-02/labels'
output_file = 'res/MOT17-02.txt'
video_width = 1920
video_height = 1080

os.makedirs('res', exist_ok=True)

with open(output_file, 'w') as fout:
    for label_file in sorted(glob.glob(os.path.join(input_dir, '*.txt'))):
        frame = int(os.path.basename(label_file).split('.')[0]) + 1
        with open(label_file, 'r') as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue

                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                track_id = int(parts[6])

                x = x_center - width / 2
                y = y_center - height / 2

                fout.write(f"{frame}, {track_id}, {x*video_width:.2f}, {y*video_height:.2f}, {width*video_width:.2f}, {height*video_height:.2f}, 1.0, -1, -1, -1\n")

print(f"✅ Etapa 2 completada: archivo MOT generado en {output_file}")

# === Archivos ===
gt_file = '/home/jetson/Documents/MOT17/train/MOT17-02-DPM/gt/gt.txt'
res_file = 'res/MOT17-02.txt'

# === Timestamp ===
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = f'resultados_{timestamp}.csv'
png_path = f'resultados_{timestamp}.png'

# === Leer ground truth ===
gt = pd.read_csv(gt_file, header=None)
gt.columns = ['FrameId', 'Id', 'X', 'Y', 'W', 'H', 'Conf', 'Class', 'Vis']
gt = gt[gt['Class'] == 1]

# === Leer resultados del tracker ===
res = pd.read_csv(res_file, header=None)
res.columns = ['FrameId', 'Id', 'X', 'Y', 'W', 'H', 'Conf', 'Unk1', 'Unk2', 'Unk3']

# === Evaluación MOT ===
acc = mm.MOTAccumulator(auto_id=True)

for frame in sorted(gt['FrameId'].unique()):
    gt_frame = gt[gt['FrameId'] == frame]
    res_frame = res[res['FrameId'] == frame]

    gt_boxes = list(zip(gt_frame['X'], gt_frame['Y'], gt_frame['W'], gt_frame['H']))
    res_boxes = list(zip(res_frame['X'], res_frame['Y'], res_frame['W'], res_frame['H']))

    dist = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)
    acc.update(gt_frame['Id'].tolist(), res_frame['Id'].tolist(), dist)

# === Métricas + CSV ===
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='MOT17-02')
summary_rounded = summary.round(3)
summary_rounded.to_csv(csv_path)
print(f"✅ CSV de métricas guardado en: {csv_path}")

# === Graficar métricas clave ===
metricas_clave = ['idf1', 'idp', 'idr', 'recall', 'precision', 'mota', 'motp']
metricas = summary_rounded.loc['MOT17-02'].reindex(metricas_clave) * 100

plt.figure(figsize=(10, 6))
bars = plt.bar(metricas.index.str.upper(), metricas.values, color='skyblue')
plt.ylabel('Porcentaje (%)')
plt.ylim(0, 100)
plt.title(f'Métricas de Evaluación - MOT17-02 ({timestamp})')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(png_path)
plt.show()

#################################### SEgundo grafico
# === GRAFICAR MÉTRICAS DE CONSUMO (si el CSV existe) ===
csv_consumo = 'personas_rendimiento.csv'  # Cambiá por el que estés usando
if os.path.exists(csv_consumo):
    df = pd.read_csv(csv_consumo)
    consumo_png = f'consumo_{timestamp}.png'

    plt.figure(figsize=(12, 6))
    plt.plot(df['imagen'], df['gpu_mem_used_MB'], label='GPU Mem (MB)', marker='o')
    plt.plot(df['imagen'], df['mem_reserved_MB'], label='Mem Reservada (MB)', marker='x')
    plt.plot(df['imagen'], df['power'], label='Potencia (W)', marker='^')
    plt.xlabel('Imagen')
    plt.ylabel('Consumo')
    plt.title(f'Métricas de Consumo - {csv_consumo}')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(consumo_png)
    plt.show()
    print(f"✅ Gráfico de consumo guardado en: {consumo_png}")
else:
    print("⚠️ No se encontró el archivo de consumo:", csv_consumo)





print(f"✅ Gráfico guardado en: {png_path}")