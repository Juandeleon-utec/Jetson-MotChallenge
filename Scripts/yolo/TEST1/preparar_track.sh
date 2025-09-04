#!/bin/bash

# === Configuración básica ===
GT_BASE="/home/jetson/Documents/MOT17/train"     # Carpeta original donde están tus datos GT
TRACKER_NAME="mi_tracker"                         # Cambia esto al nombre de tu tracker
SEQS=("MOT17-02-DPM")                             # Lista de secuencias a evaluar

# === Carpetas de TrackEval ===
GT_FOLDER="/home/jetson/Documents/TrackEval/data/gt/mot_challenge/MOT17/train"
TRACKER_FOLDER="/home/jetson/Documents/TrackEval/data/trackers/mot_challenge/MOT17/train"

for SEQ in "${SEQS[@]}"; do
    echo "Preparando $SEQ..."

    ORIG_GT="$GT_BASE/$SEQ"
    DEST_GT="$GT_FOLDER/$SEQ"
    DEST_TRACKER="$TRACKER_FOLDER/$SEQ"

    # Crear carpetas
    mkdir -p "$DEST_GT/gt"
    mkdir -p "$DEST_TRACKER"

    # Copiar ground truth
    if [[ -f "$ORIG_GT/gt/gt.txt" ]]; then
        cp "$ORIG_GT/gt/gt.txt" "$DEST_GT/gt/gt.txt"
    else
        echo "⚠️  Ground truth no encontrado para $SEQ"
    fi

    # Copiar detecciones (ajustá el nombre según tu archivo)
    if [[ -f "detecciones_motchallenge.txt" ]]; then
        cp "detecciones_motchallenge.txt" "$DEST_TRACKER/track.txt"
    else
        echo "⚠️  detecciones_motchallenge.txt no encontrado"
    fi

    # Crear seqinfo.ini (ajustar valores si son diferentes)
    cat <<EOF > "$DEST_GT/seqinfo.ini"
[Sequence]
name=$SEQ
imDir=img1
frameRate=30
seqLength=600
imWidth=1920
imHeight=1080
EOF

    echo "✅ $SEQ preparado correctamente"
done
