# src/neural_network.py
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, RandomRotation, RandomZoom, RandomTranslation, RandomContrast, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from config import MODEL_PATH, IMG_SIZE, FOLDERS
from data_acquisition import get_binary_roi, get_projections

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

feature_extractor = None
SIGNATURE_DATA = {} 

def load_ai():
    global feature_extractor
    if not TF_AVAILABLE: return
    try:
        if os.path.exists(MODEL_PATH):
            print("[SYSTEM] Încărcare model Custom Nano...")
            custom_model = load_model(MODEL_PATH)
            # Încercăm să extragem layer-ul dens penultim
            try:
                feature_extractor = Model(inputs=custom_model.input, outputs=custom_model.get_layer("embedding_layer").output)
            except:
                # Dacă nu reușim, folosim output-ul direct (mai puțin ideal, dar funcționează)
                feature_extractor = custom_model
        else:
            print("[SYSTEM] Model inexistent. Așteptare antrenare...")
            feature_extractor = None
        
        print("[SYSTEM] AI Engine Online.")
        reload_cache()
    except Exception as e: 
        print(f"[AI Error] {e}")

def reload_cache():
    global SIGNATURE_DATA
    SIGNATURE_DATA = {}
    train_path = FOLDERS["train"]
    if not os.path.exists(train_path) or feature_extractor is None: return

    for student in os.listdir(train_path):
        s_dir = os.path.join(train_path, student)
        if not os.path.isdir(s_dir): continue
        vectors = []
        geo_refs = []
        for f in os.listdir(s_dir):
            if f.endswith('.png'):
                img = cv2.imread(os.path.join(s_dir, f))
                if img is not None:
                    img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    if len(img_res.shape) == 2: img_res = cv2.cvtColor(img_res, cv2.COLOR_GRAY2RGB)
                    elif img_res.shape[2] == 3: img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                    batch = preprocess_input(img_res.astype(np.float32))
                    
                    try:
                        vec = feature_extractor.predict(np.expand_dims(batch, 0), verbose=0).flatten()
                        vectors.append(vec)
                    except: pass
                    
                    roi, ar = get_binary_roi(img)
                    if roi is not None:
                        proj = get_projections(roi)
                        geo_refs.append((roi, ar, proj))
        if vectors:
            SIGNATURE_DATA[student] = {"vectors": vectors, "geo_refs": geo_refs}

def load_dataset_from_folder(folder_path):
    images = []
    labels = []
    if not os.path.exists(folder_path): return np.array([]), np.array([])
    
    students = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    
    for student in students:
        s_path = os.path.join(folder_path, student)
        for img_name in os.listdir(s_path):
            if img_name.endswith('.png'):
                img = cv2.imread(os.path.join(s_path, img_name))
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(student)
    
    X = np.array(images)
    if len(X) > 0:
        X = preprocess_input(X.astype(np.float32))
    return X, labels

def train_custom_model():
    if not TF_AVAILABLE: return "TensorFlow lipsă."
    
    print("[TRAIN] Încărcare date pentru Custom Network...")
    X_train, y_train_labels = load_dataset_from_folder(FOLDERS["train"])
    X_val, y_val_labels = load_dataset_from_folder(FOLDERS["val"])
    
    if len(X_train) == 0: return "Nu există date în folderul train!"

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_labels)
    try:
        y_val_enc = le.transform(y_val_labels)
    except:
        le.fit(list(y_train_labels) + list(y_val_labels))
        y_train_enc = le.transform(y_train_labels)
        y_val_enc = le.transform(y_val_labels)

    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
    y_val_cat = to_categorical(y_val_enc, num_classes=num_classes)

    # --- ARHITECTURA "NANO" ORIGINALĂ ---
    # Concepută special pentru dataset-uri foarte mici (<20 imagini/clasă)
    
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # 1. Augmentare Agresivă
        RandomRotation(0.1),       # Rotire +/- 10%
        RandomZoom(0.1),           # Zoom +/- 10%
        RandomTranslation(0.1, 0.1), # Deplasare
        RandomContrast(0.2),       # Schimbare contrast (simulează pixuri diferite)

        # 2. Extragere Trăsături (Foarte simplu!)
        # Strat 1: Filtre puține (16) + Regularizare L2 (evită memorarea)
        Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        
        # Strat 2: Puțin mai complex (32)
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        
        # Strat 3: Maximul de complexitate permis
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        
        # 3. Clasificare cu Dropout masiv
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), name="embedding_layer"),
        Dropout(0.6), # Oprește 60% din neuroni random (Forțează rețeaua să fie robustă)
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Learning Rate Standard
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"[TRAIN] Start antrenare Nano-Network (Original)...")
    
    # Antrenăm mai mult (60 epoci) cu batch_size mic (4) pentru a face update-uri dese
    history = model.fit(X_train, y_train_cat, epochs=60, batch_size=4, validation_data=(X_val, y_val_cat), verbose=1)
    
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        plt.title('Grafic Convergență (Original Nano-Model)')
        plt.xlabel('Epoci')
        plt.ylabel('Eroare (Loss)')
        plt.legend()
        plt.grid(True)
        if not os.path.exists('docs'): os.makedirs('docs')
        plt.savefig(os.path.join('docs', 'loss_curve.png'))
        plt.close()
    except Exception as e: print(f"[WARN] Eroare grafic: {e}")

    model.save(MODEL_PATH)
    load_ai()
    return f"Succes! Model Original Antrenat."

def get_similarity_score(t1, t2):
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return float(np.dot(t1, t2) / (norm1 * norm2))