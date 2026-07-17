import os
import glob
import gzip
import shutil

cache_dir = "cloud_models_cache"

# Mostrar todos los archivos en la cache
all_files = glob.glob(os.path.join(cache_dir, "**", "*"), recursive=True)
print(f"Total archivos en cache: {len(all_files)}")
for f in all_files:
    if os.path.isfile(f):
        print(f)
