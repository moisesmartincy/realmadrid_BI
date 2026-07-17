import gzip
import joblib
import os

modelos = [
    "modelo_asistencia_entrenado",
    "modelos_entrenados",
    "modelo_total_productos_ganados_entrenado",
]

cache = "cloud_models_cache"

for m in modelos:
    enc_path = os.path.join(cache, m, "feature_encoders.pkl.gz")
    if os.path.exists(enc_path):
        enc = joblib.load(enc_path)
        print(f"\n=== {m} === ENCODERS:")
        for k, v in enc.items():
            print(f"  {k}: classes = {list(v.classes_)}")
    
    lbl_path = os.path.join(cache, m, "label_encoder_target.pkl.gz")
    if os.path.exists(lbl_path):
        lbl = joblib.load(lbl_path)
        print(f"  TARGET ENCODER: classes = {list(lbl.classes_)}")
