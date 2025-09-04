import os
import cv2
from shutil import copy2

mot_path = "/ruta/a/MOT17/train"
output_img_dir = "dataset/images/train"
output_lbl_dir = "dataset/labels/train"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

sequences = [d for d in os.listdir(mot_path) if os.path.isdir(os.path.join(mot_path, d))]

for seq in sequences:
    seq_path = os.path.join(mot_path, seq)
    img_dir = os.path.join(seq_path, "img1")
    gt_file = os.path.join(seq_path, "gt", "gt.txt")

    # Cargar dimensiones de imágenes
    first_img_path = os.path.join(img_dir, "000001.jpg")
    img = cv2.imread(first_img_path)
    if img is None:
        continue
    ih, iw = img.shape[:2]

    # Parsear anotaciones
    with open(gt_file, "r") as f:
        for line in f:
            frame, obj_id, x, y, w, h, cls_id, *_ = line.strip().split(",")
            frame = int(frame)
            cls_id = int(cls_id)
            if cls_id != 1:  # Solo personas si estás interesado en clase 1
                continue

            x, y, w, h = map(float, (x, y, w, h))

            # Normalizar
            x_center = (x + w / 2) / iw
            y_center = (y + h / 2) / ih
            w_norm = w / iw
            h_norm = h / ih

            # Nombre único de imagen y etiqueta
            frame_name = f"{seq}_{frame:06d}"
            img_name = f"{frame_name}.jpg"
            label_name = f"{frame_name}.txt"

            # Copiar imagen (una sola vez)
            src_img_path = os.path.join(img_dir, f"{frame:06d}.jpg")
            dst_img_path = os.path.join(output_img_dir, img_name)
            if not os.path.exists(dst_img_path):  # copiar solo si no existe
                copy2(src_img_path, dst_img_path)

            # Escribir etiqueta
            label_path = os.path.join(output_lbl_dir, label_name)
            with open(label_path, 'a') as out_f:
                out_f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
