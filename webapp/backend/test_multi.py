import gzip
import json
import joblib

path = "c:/Users/KEVIN ZAPANA/Desktop/csvModelosMachine/webapp/backend/cloud_models_cache/modelo_top_productos_entrenado"
try:
    with gzip.open(path + "/metadata.json.gz", "rt", encoding="utf-8") as f:
        meta = json.load(f)
        print("METADATA:", meta)
    
    enc = joblib.load(path + "/feature_encoders.pkl.gz")
    print("ENCODERS:", list(enc.keys()))
    
except Exception as e:
    print(e)
