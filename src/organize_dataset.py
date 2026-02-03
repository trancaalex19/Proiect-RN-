# src/organize_dataset.py
import os
import shutil
import random
import numpy as np

# Căile hardcodate pentru siguranță
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Folderul Proiect RN
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

def organize():
    # 1. Creăm folderele dacă nu există
    for d in [VAL_DIR, TEST_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 2. Citim studenții din Train
    if not os.path.exists(TRAIN_DIR):
        print("Eroare: Nu găsesc folderul 'data/train'!")
        return

    students = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    print(f"Am găsit {len(students)} studenți. Încep redistribuirea fișierelor (70% Train, 15% Val, 15% Test)...")

    total_moved_val = 0
    total_moved_test = 0

    for student in students:
        s_train_path = os.path.join(TRAIN_DIR, student)
        s_val_path = os.path.join(VAL_DIR, student)
        s_test_path = os.path.join(TEST_DIR, student)

        # Creăm folderele studentului în Val și Test
        os.makedirs(s_val_path, exist_ok=True)
        os.makedirs(s_test_path, exist_ok=True)

        # Luăm toate imaginile
        images = [f for f in os.listdir(s_train_path) if f.endswith('.png')]
        random.shuffle(images) # Le amestecăm ca să fie random

        count = len(images)
        if count == 0: continue

        # Calculăm câte mutăm
        n_val = int(count * 0.15)
        n_test = int(count * 0.15)
        
        # Selectăm fișierele
        files_for_val = images[:n_val]
        files_for_test = images[n_val : n_val + n_test]

        # Mutăm fizic fișierele
        for f in files_for_val:
            shutil.move(os.path.join(s_train_path, f), os.path.join(s_val_path, f))
            total_moved_val += 1
            
        for f in files_for_test:
            shutil.move(os.path.join(s_train_path, f), os.path.join(s_test_path, f))
            total_moved_test += 1

        print(f" -> {student}: {count} total => {len(images) - n_val - n_test} Train | {n_val} Val | {n_test} Test")

    print("="*50)
    print(f"GATA! Au fost mutate {total_moved_val} poze în 'val' și {total_moved_test} poze în 'test'.")
    print("Verifică folderele acum!")

if __name__ == "__main__":
    organize()
    