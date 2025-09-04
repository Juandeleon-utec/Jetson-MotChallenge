import os
import zipfile

# Lista de secuencias a procesar
SECUENCIAS = [
    "MOT17-01-DPM", "MOT17-01-FRCNN", "MOT17-01-SDP",
    "MOT17-03-DPM", "MOT17-03-FRCNN", "MOT17-03-SDP",
    "MOT17-06-DPM", "MOT17-06-FRCNN", "MOT17-06-SDP",
    "MOT17-07-DPM", "MOT17-07-FRCNN", "MOT17-07-SDP",
    "MOT17-08-DPM", "MOT17-08-FRCNN", "MOT17-08-SDP",
    "MOT17-12-DPM", "MOT17-12-FRCNN", "MOT17-12-SDP",
    "MOT17-14-DPM", "MOT17-14-FRCNN", "MOT17-14-SDP",
     "MOT17-02-DPM", "MOT17-02-FRCNN", "MOT17-02-SDP",
    "MOT17-04-DPM", "MOT17-04-FRCNN", "MOT17-04-SDP",
    "MOT17-05-DPM", "MOT17-05-FRCNN", "MOT17-05-SDP",
    "MOT17-09-DPM", "MOT17-09-FRCNN", "MOT17-09-SDP",
    "MOT17-10-DPM", "MOT17-10-FRCNN", "MOT17-10-SDP",
    "MOT17-11-DPM", "MOT17-11-FRCNN", "MOT17-11-SDP",
    "MOT17-13-DPM", "MOT17-13-FRCNN", "MOT17-13-SDP"
]

# Crear carpeta de salida
OUTPUT_DIR = "corregidos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def corregir_formato_mot(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for linea in f_in:
            elementos = linea.strip().split()
            if len(elementos) >= 10:
                elementos = elementos[:10]
                f_out.write(','.join(elementos) + '\n')

# Procesar archivos
for secuencia in SECUENCIAS:
    input_filename = f"{secuencia}.txt"
    output_filename = os.path.join(OUTPUT_DIR, f"{secuencia}.txt")

    if os.path.exists(input_filename):
        print(f"✔ Procesando {input_filename}")
        corregir_formato_mot(input_filename, output_filename)
    else:
        print(f"⚠ Archivo no encontrado: {input_filename}")

# Crear archivo ZIP con todos los corregidos
ZIP_FILENAME = "resultados_motchallenge.zip"
with zipfile.ZipFile(ZIP_FILENAME, 'w') as zipf:
    for archivo in os.listdir(OUTPUT_DIR):
        if archivo.endswith(".txt"):
            filepath = os.path.join(OUTPUT_DIR, archivo)
            zipf.write(filepath, arcname=archivo)  # Guardar en raíz del zip
print(f"\n✅ Archivo ZIP generado: {ZIP_FILENAME}")
