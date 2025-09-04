import sys
import re
import pandas as pd
import matplotlib.pyplot as plt

# === MÉTRICAS A EXTRAER ===
metricas_deseadas = [
    "MOTA", "IDF1", "HOTA", "MT", "ML", "FP", "FN", "CLR_Re", "CLR_Pr",
    "AssA", "DetA", "AssRe", "AssPr", "DetRe", "DetPr", "LocA", "FAF",
    "IDSW", "Frag", "Hz"
]

# === FUNCION PARA EXTRAER METRICAS COMBINED DE UN ARCHIVO ===
def extraer_metricas(filename):
    data = {"Archivo": filename}
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    combined_found = False
    for line in lines:
        if line.strip().startswith("COMBINED"):
            combined_found = True
            partes = re.split(r'\s{2,}', line.strip())
            if len(partes) >= len(metricas_deseadas) + 1:
                for i, key in enumerate(metricas_deseadas):
                    try:
                        val = partes[i + 1]
                        if val.replace('.', '', 1).isdigit():
                            data[key] = float(val)
                        else:
                            data[key] = val
                    except:
                        data[key] = None
            break

    return data if combined_found else None

# === PROCESAR ARCHIVOS PASADOS POR ARGUMENTO ===
archivos = sys.argv[1:]
if not archivos:
    print("Uso: python3 comparar.py archivo1.txt archivo2.txt ...")
    sys.exit(1)

datos = []
for archivo in archivos:
    resultado = extraer_metricas(archivo)
    if resultado:
        datos.append(resultado)
    else:
        print(f"[!] No se encontró sección COMBINED en: {archivo}")

# === CREAR DATAFRAME Y EXPORTAR CSV ===
df = pd.DataFrame(datos)
df.rename(columns={
    "CLR_Re": "Rcll",
    "CLR_Pr": "Prcn",
    "IDSW": "ID Sw."
}, inplace=True)

df.to_csv("comparacion.csv", index=False)
print("✅ Archivo comparacion.csv generado.")

# === GENERAR GRAFICO DE BARRAS PARA METRICAS <= 100 ===
df_plot = df.set_index("Archivo")
cols_plot = [col for col in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[col]) and df_plot[col].max() <= 100]

if cols_plot:
    df_plot[cols_plot].T.plot(kind="bar", figsize=(14, 7), rot=45)
    plt.title("Comparación de métricas COMBINED")
    plt.ylabel("Valor")
    plt.grid(axis='y')
    plt.legend(title="Archivo")
    plt.tight_layout()
    plt.savefig("comparacion.png")
    print("✅ Gráfico comparacion.png generado.")
else:
    print("⚠️ No hay métricas ≤ 100 para graficar.")
