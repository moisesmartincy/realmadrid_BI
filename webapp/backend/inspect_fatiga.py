import gzip
import joblib
import os

cache = "cloud_models_cache"
m = "modelo_fatiga_entrenado"

enc_path = os.path.join(cache, m, "feature_encoders.pkl.gz")
enc = joblib.load(enc_path)
print(f"=== {m} === ENCODERS:")
for k, v in enc.items():
    print(f"  {k}: classes = {list(v.classes_)}")
