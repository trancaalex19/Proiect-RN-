# config.py
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

BASE_DIR = os.path.join(ROOT_DIR, "data")
# IMPORTANT: Salvăm modelul în folderul models
MODEL_PATH = os.path.join(ROOT_DIR, "models", "semnatura_model.h5")

FOLDERS = {
    "raw_sig": os.path.join(BASE_DIR, "raw_sig"),
    "raw_list": os.path.join(BASE_DIR, "raw_list"),
    "processed_sig": os.path.join(BASE_DIR, "processed_sig"),
    "processed_list": os.path.join(BASE_DIR, "processed_list"),
    
    # AICI SUNT NOILE CĂI:
    "train": os.path.join(BASE_DIR, "train"),
    "val": os.path.join(BASE_DIR, "val"),   # <--- NOU
    "test": os.path.join(BASE_DIR, "test")  # <--- NOU
}

IMG_SIZE = 160

def init_env():
    for f in FOLDERS.values():
        os.makedirs(f, exist_ok=True)
    # Creăm și folderul models dacă nu există
    os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)