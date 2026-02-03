import os
import cv2
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Importăm variabilele tale de configurare (IMG_SIZE, FOLDERS)
from config import IMG_SIZE, FOLDERS

def incarca_datele_de_antrenament():
    """
    Această funcție preia logica de încărcare a imaginilor din neural_network.py
    pentru a nu duplica codul manual.
    """
    print("[SETUP] Se încarcă imaginile pentru experimente...")
    train_dir = FOLDERS["train"]
    
    if not os.path.exists(train_dir):
        print("Eroare: Folderul 'train' nu există!")
        return None, None, None

    students = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if len(students) < 2:
        print("Eroare: Ai nevoie de minim 2 studenți pentru a rula experimente.")
        return None, None, None

    images = []
    labels = []

    for student in students:
        s_path = os.path.join(train_dir, student)
        for img_name in os.listdir(s_path):
            if img_name.endswith('.png'):
                try:
                    img = cv2.imread(os.path.join(s_path, img_name))
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(student)
                except Exception as e:
                    print(f"Eroare la citirea imaginii {img_name}: {e}")

    if len(images) < 10:
        print("Eroare: Prea puține imagini pentru un experiment valid.")
        return None, None, None

    X = np.array(images)
    X = preprocess_input(X.astype(np.float32))

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(np.unique(y_encoded))
    y_cat = to_categorical(y_encoded, num_classes=num_classes)

    print(f"[SETUP] Date încărcate: {len(X)} imagini, {num_classes} clase.")
    return X, y_cat, num_classes

def construieste_model(num_classes, learning_rate):
    """
    Configurație: Antrenare de la ZERO (weights=None) + Toate straturile deblocate.
    """
    # 1. weights=None (Fără ImageNet)
    base = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # 2. IMPORTANT: Deblocăm straturile pentru a permite învățarea de la zero
    for layer in base.layers: 
        layer.trainable = True
        
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) 
    x = Dense(128, activation='relu', name="embedding_layer")(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    X, y, num_classes = incarca_datele_de_antrenament()
    if X is None:
        return

    # Testăm 3 rate de învățare relevante
    learning_rates_de_testat = [0.0001, 0.0005, 0.001]
    
    rezultate_experimente = []

    print("\n[START] Începe rularea experimentelor de optimizare (Antrenare de la zero)...")

    for lr in learning_rates_de_testat:
        print(f"\n--- Experiment: Learning Rate = {lr} ---")
        
        model = construieste_model(num_classes, learning_rate=lr)
        
        start_time = time.time()
        
        # Am crescut la 25 epoci pentru a da timp modelului să învețe ceva de la zero
        history = model.fit(X, y, epochs=25, batch_size=8, validation_split=0.2, verbose=0)
        
        end_time = time.time()
        durata = end_time - start_time
        
        max_val_acc = max(history.history['val_accuracy'])
        min_val_loss = min(history.history['val_loss'])
        final_train_acc = history.history['accuracy'][-1]
        
        print(f"   -> Rezultat: Val Acc: {max_val_acc:.4f} | Train Acc: {final_train_acc:.4f} | Durata: {durata:.2f}s")

        rezultate_experimente.append({
            'Experiment_ID': len(rezultate_experimente) + 1,
            'Learning_Rate': lr,
            'Train_Accuracy': round(final_train_acc, 4),
            'Validation_Accuracy': round(max_val_acc, 4),
            'Validation_Loss': round(min_val_loss, 4),
            'Durata_Secunde': round(durata, 2),
            'Epoci': 25
        })

    print("\n[INFO] Salvare rezultate în CSV...")
    
    folder_rezultate = 'results'
    os.makedirs(folder_rezultate, exist_ok=True)
    
    df = pd.DataFrame(rezultate_experimente)
    path_csv = os.path.join(folder_rezultate, 'optimization_experiments.csv')
    
    df.to_csv(path_csv, index=False)
    
    print("="*50)
    print(f"SUCCES! Fișierul a fost generat aici:\n{os.path.abspath(path_csv)}")
    print("="*50)
    print(df)

if __name__ == "__main__":
    main()