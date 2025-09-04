import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === Leer detecciones del archivo ===
detecciones = []
with open("detecciones_motchallenge.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 7:
            frame = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            detecciones.append([frame, obj_id, x, y, w, h, conf])

# === Filtrar solo un frame para visualizar (por ejemplo, frame 1) ===
frame_a_graficar = 1
detecciones_frame = [d for d in detecciones if d[0] == frame_a_graficar]

# === Dibujar los bounding boxes ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 1920)
ax.set_ylim(1080, 0)
ax.set_title(f"Detecciones MOTChallenge - Frame {frame_a_graficar}")
ax.set_xlabel("Eje X")
ax.set_ylabel("Eje Y")

for _, obj_id, x, y, w, h, conf in detecciones_frame:
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y - 10, f"ID:{obj_id} ({conf:.2f})", color='blue', fontsize=10)

plt.tight_layout()
plt.savefig("grafico_detecciones.png")
plt.show()
