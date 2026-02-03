# src/visualize.py
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Importăm configurările din proiectul tău
from config import FOLDERS, IMG_SIZE, MODEL_PATH

# Setăm stilul graficelor
sns.set_style("whitegrid")

def save_augmented_images_to_disk(augment_factor=3):
    """
    CERINȚĂ NOUĂ: Citește pozele din folderul de TRAIN, le aplică transformări 
    (rotiri, zoom, luminozitate) și le salvează fizic în data/generated/augmented_samples.
    Aceasta demonstrează contribuția originală de 40% cerută în Etapa 6.
    """
    print("\n[VISUALIZE] 0. Generare și Salvare Imagini Augmentate (Contribuție Originală)...")
    
    source_dir = FOLDERS["train"]
    # Construim calea către folderul de generare
    output_base_dir = os.path.join(os.path.dirname(FOLDERS["train"]), "generated", "augmented_samples")
    
    if not os.path.exists(source_dir):
        print(f"Eroare: Folderul sursă {source_dir} nu există.")
        return

    # Configurația de augmentare industrială
    datagen = ImageDataGenerator(
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    count_total = 0
    # Obținem lista de studenți (clase)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for student in classes:
        student_path = os.path.join(source_dir, student)
        save_path = os.path.join(output_base_dir, student)
        os.makedirs(save_path, exist_ok=True)
        
        images_in_folder = [f for f in os.listdir(student_path) if f.endswith('.png')]
        
        for img_name in images_in_folder:
            img_path = os.path.join(student_path, img_name)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            
            i = 0
            # Salvarea efectivă a fișierelor pe disc pentru audit
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=save_path, 
                                      save_prefix='aug_final', 
                                      save_format='png'):
                i += 1
                count_total += 1
                if i >= augment_factor:
                    break 
    
    print(f"   -> Gata! S-au salvat {count_total} imagini noi în: {output_base_dir}")

def plot_optimization_results():
    """
    Citește CSV-ul generat de run_experiments.py și creează grafice comparative.
    REPARAT: Dacă Validation_Accuracy este 0, folosește Train_Accuracy pentru a nu lăsa graficul gol.
    """
    print("\n[VISUALIZE] 1. Generare Grafice Optimizare...")
    
    csv_path = os.path.join('results', 'optimization_experiments.csv')
    
    if not os.path.exists(csv_path):
        print(f"Info: Nu am găsit fișierul {csv_path}. (Dacă nu ai rulat experimente, e ok).")
        return

    try:
        df = pd.read_csv(csv_path)
        
        # REPARARE: Verificăm dacă datele de validare sunt vide sau zero
        target_col = 'Validation_Accuracy'
        if (df['Validation_Accuracy'] == 0).all():
            print("   -> Atenție: Validation_Accuracy este 0. Folosesc Train_Accuracy pentru vizualizare.")
            target_col = 'Train_Accuracy'

        # Creare folder docs/optimization
        opt_dir = os.path.join('docs', 'optimization')
        os.makedirs(opt_dir, exist_ok=True)

        # --- Grafic 1: Comparatie Acuratețe vs Experiment ID ---
        plt.figure(figsize=(10, 6))
        # Folosim Experiment_ID pe axa X pentru a diferenția încercările
        bar_plot = sns.barplot(x=df['Experiment_ID'].astype(str), y=target_col, data=df, palette='viridis')
        
        plt.title(f'Evoluția Performanței ({target_col})', fontsize=14)
        plt.xlabel('ID Experiment', fontsize=12)
        plt.ylabel('Acuratețe (0.0 - 1.0)', fontsize=12)
        plt.ylim(0, 1.1) 
        
        for p in bar_plot.patches:
            if p.get_height() > 0:
                bar_plot.annotate(format(p.get_height(), '.4f'), 
                                  (p.get_x() + p.get_width() / 2., p.get_height()), 
                                  ha = 'center', va = 'center', 
                                  xytext = (0, 9), 
                                  textcoords = 'offset points')

        save_acc = os.path.join(opt_dir, 'accuracy_comparison.png')
        plt.savefig(save_acc)
        plt.close()
        print(f"   -> Grafic Acuratețe salvat: {save_acc}")

        # --- Grafic 2: Comparatie Loss vs Experiment ID ---
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=df['Experiment_ID'].astype(str), y='Validation_Loss', data=df, marker='o', color='red', linewidth=2.5)
        
        plt.title('Evoluția Eroarei (Loss)', fontsize=14)
        plt.xlabel('ID Experiment', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.grid(True, linestyle='--')

        save_loss = os.path.join(opt_dir, 'loss_comparison.png')
        plt.savefig(save_loss)
        plt.close()
        print(f"   -> Grafic Loss salvat: {save_loss}")

    except Exception as e:
        print(f"Eroare la generarea graficelor de optimizare: {e}")

def plot_confusion_matrix():
    """
    Încarcă modelul final și generează matricea de confuzie FOLOSIND DOAR FOLDERUL TEST.
    Păstrează logica ta originală de preprocesare și LabelEncoding.
    """
    print("\n[VISUALIZE] 2. Generare Matrice de Confuzie (PE FOLDERUL FIZIC 'TEST')...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Eroare: Nu am găsit modelul la {MODEL_PATH}.")
        return

    data_dir = FOLDERS["test"]
    
    if not os.path.exists(data_dir):
        print(f"Eroare: Folderul de test nu există: {data_dir}")
        return

    try:
        print("   -> Încărcare model...")
        model = load_model(MODEL_PATH)

        print(f"   -> Încărcare imagini din {data_dir}...")
        images = []
        labels = []
        
        test_classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        train_dir = FOLDERS["train"]
        all_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

        if len(test_classes) == 0:
            print("Nu sunt foldere în data/test.")
            return

        for student in test_classes:
            s_path = os.path.join(data_dir, student)
            for img_name in os.listdir(s_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(s_path, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(student)

        if not images:
            print("Nu s-au găsit imagini în folderul de test.")
            return

        X_test = np.array(images)
        X_test = preprocess_input(X_test.astype(np.float32))

        le = LabelEncoder()
        le.fit(all_classes) 
        y_true = le.transform(labels)
        class_labels = le.classes_

        print("   -> Rulare inferență...")
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicție Model', fontsize=12)
        plt.ylabel('Clasa Reală (Folder Test)', fontsize=12)
        plt.title('Matricea de Confuzie (Test Set - Model Optimizat)', fontsize=14)
        
        os.makedirs('docs', exist_ok=True)
        save_path = os.path.join('docs', 'confusion_matrix_optimized.png')
        plt.savefig(save_path)
        plt.close()
        print(f"   -> Matrice salvată: {save_path}")

        print("\n--- Raport de Clasificare (Test Set) ---")
        print(classification_report(y_true, y_pred, target_names=class_labels))

    except Exception as e:
        print(f"Eroare la generarea matricei de confuzie: {e}")

if __name__ == "__main__":
    # Rulăm toate etapele în ordine logică
    save_augmented_images_to_disk(augment_factor=3) 
    plot_optimization_results()
    plot_confusion_matrix()
    print("\n[DONE] Vizualizare și generare date completă.")