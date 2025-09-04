import os
import glob

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
