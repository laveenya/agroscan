"""DVC stage 1: unpack the prepared dataset zip into data/prepared."""
import os
import shutil
import zipfile

from src.config import load_params

p = load_params()["data"]
shutil.rmtree(p["prepared_dir"], ignore_errors=True)
os.makedirs(p["prepared_dir"], exist_ok=True)
with zipfile.ZipFile(p["zip_path"]) as z:
    z.extractall(p["prepared_dir"])
print("unpacked sets:", sorted(os.listdir(p["prepared_dir"])))
