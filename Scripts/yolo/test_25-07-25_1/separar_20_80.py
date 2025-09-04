import os
import random
import shutil

# === Directorios base ===
base_path = "dataset"
images_path = os.path.join(base_path, "images/train")
labels_path = os.path.join(base_path, "labels/train")

# === Nuevos directorios destino ===
new_images_train = os.path.join(base_path, "images/train_split")
new_images_val   = os.path.join(base_path, "images/val")
new_labels_train = os.path.join(base_path, "labels/train_split")
new_labels_val   = os.path.join(base_path, "labels/val")

for d in [new_images_train, new_images_val, new_labels_train, new_labels_val]:
    os.makedirs(d, exist_ok=True)

# === Obtener todas las imágenes ===
image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg")]
random.shuffle(image_files)

split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# === Función para copiar imágenes y etiquetas ===
def copiar_set(files, img_src, lbl_src, img_dst, lbl_dst):
    for img_file in files:
        label_file = img_file.replace(".jpg", ".txt")

        # Copiar imagen
        shutil.copy(os.path.join(img_src, img_file), os.path.join(img_dst, img_file))

        # Copiar etiqueta si existe
        label_src_path = os.path.join(lbl_src, label_file)
        if os.path.exists(label_src_path):
            shutil.copy(label_src_path, os.path.join(lbl_dst, label_file))

# === Ejecutar copia ===
copiar_set(train_files, images_path, labels_path, new_images_train, new_labels_train)
copiar_set(val_files,   images_path, labels_path, new_images_val,   new_labels_val)

print(f"🔁 División completada: {len(train_files)} entrenamiento / {len(val_files)} validación")
