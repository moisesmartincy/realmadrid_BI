import gzip
import json
import os

modelos = [
    "modelo_asistencia_entrenado",
    "modelo_fatiga_entrenado",
    "modelo_jugador_entrenado",
    "modelos_entrenados",
    "modelo_top_productos_entrenado",
    "modelo_total_productos_ganados_entrenado",
]

cache = "cloud_models_cache"

for m in modelos:
    col_path = os.path.join(cache, m, "columnas_features.json.gz")
    meta_path = os.path.join(cache, m, "metadata.json.gz")
    
    if os.path.exists(col_path):
        with gzip.open(col_path, 'rt', encoding='utf-8') as f:
            cols = json.load(f)
        print(f"\n=== {m} === FEATURES ({len(cols)}):")
        print(cols)
    
    if os.path.exists(meta_path):
        with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
            meta = json.load(f)
        print(f"  METADATA: {meta}")
