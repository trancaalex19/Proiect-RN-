
# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale (SVAS)

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Trancă Alexandru-Constantin
**Proiect:** SVAS - Sistem Avansat de Verificare a Semnăturilor
**Link Repository GitHub:** [https://github.com/trancaalex19/Proiect-RN-.git]
**Data:** 20.01.2026

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3** pentru proiectul **SVAS**. Obiectivul este transformarea documentelor brute (PDF-uri cu liste de semnături) într-un set de date structurat (imagini PNG normalizate), pregătit pentru antrenarea unei rețele neuronale convoluționale (bazată pe MobileNetV2).

---

## 1. Structura Repository-ului (Adaptată Proiectului)

Conform codului sursă (`svas_web.py`) și a structurii de fișiere, organizarea datelor este următoarea:

```text
Proiect RN/
├── readme-Etapa3.md                  # Acest fișier
├── svas_web.py                # Codul sursă principal (Backend + UI)
├── semnatura_model.h5         # Modelul antrenat (artifact generat)
├── data/
│   ├── raw_sig/               # [Input] PDF-uri brute pentru înrolare (Training)
│   ├── raw_list/              # [Input] PDF-uri brute pentru verificare (Testing)
│   ├── processed_sig/         # [Intermediar] Semnături decupate și curățate
│   ├── processed_list/        # [Intermediar] Semnături extrase pentru verificare
│   └── train/                 # [Output Final] Dataset structurat pe clase (Foldere cu Nume_Student)
└── requirements.txt           # Dependențe (tensorflow, opencv-python, flask, pymupdf)

```

---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Date reale, digitizate.
* **Modul de achiziție:** Scanare documente PDF / Încărcare fișiere PDF (liste de prezență, formulare).
* **Procesul de extracție:** Algoritm de Computer Vision implementat în care detectează "insule" (blobs) de cerneală în pagină.(`extract_signatures_blob_mode`)

### 2.2 Caracteristicile dataset-ului

* **Tip date intrare:** Imagini raster (extrase din PDF la rezoluție înaltă factor zoom 3.0).
* **Tip date ieșire (Preprocesate):** Imagini `.png`, dimensiune fixă **160x160 pixeli**.
* **Canale:** 3 canale (RGB) – necesar pentru transfer learning cu MobileNetV2.
* **Organizare:** Clasificare supervizată. Fiecare folder din `data/train/` reprezintă o clasă (un student).

### 2.3 Descrierea caracteristicilor extrase (Features)

Pe lângă imaginea brută (pixel data), sistemul extrage și vectori biometrici geometrici pentru etapa de validare hibridă:

+---------+---------+----------------------+-------------------+
| Feature | Tip     | Sursă (Funcție)      | Detalii           |
+---------+---------+----------------------+-------------------+
| Pixels  | Tensor  | img                  | Norm. [-1, 1]     |
| Aspect  | Float   | get_binary_roi       | Raport L/H        |
| Center  | (x, y)  | center_image_by_mass | Centroid (align)  |
| Proj.   | Vector  | get_projections      | Hist. X/Y         |
| Grid    | Matrice | get_density_grid     | Densitate (4x5)   |
+---------+---------+----------------------+-------------------+

## 3. Analiza Exploratorie a Datelor (EDA)

Analiza datelor se realizează automat la înrolare. Principalele provocări identificate și tratate în cod:

### 3.1 Variabilitatea Dimensională

Semnăturile extrase au dimensiuni variate.

* **Soluție:** Resize cu păstrarea raportului de aspect (aspect ratio preservation) și padare (centrare) pe canvas de 160x160 px.

### 3.2 Zgomot și Artefacte

PDF-urile conțin adesea linii de tabel, text tipărit sau pete.

* **Filtrare:** Se elimină blob-urile prea mici (`w > 20` și `h > 15`).
* **Zona sigură:** Se ignoră ultimii 10% din pagină pentru a evita subsolurile/watermark-urile (`safe_height`).

### 3.3 Contrast Variabil

Unele semnături sunt șterse sau scanate slab.

* **Soluție:** Aplicare CLAHE (Contrast Limited Adaptive Histogram Equalization) și binarizare OTSU (`cv2.THRESH_OTSU`).

---

## 4. Preprocesarea Datelor (Implementată în `svas_web.py`)

Fluxul de preprocesare („Pipeline”) implementat în funcțiile `extract_signatures_blob_mode` și `finalize_crop` include:

### 4.1 Curățare și Extragere (Computer Vision)

1. **Conversie PDF-Imagine:** Folosind librăria `PyMuPDF` (fitz).
2. **Grayscale & Blur:** Reducerea zgomotului gaussian.
3. **Dilatare:** `cv2.dilate` pentru a uni segmentele întrerupte ale semnăturii.
4. **Contururi:** Identificarea regiunilor de interes (ROI).

### 4.2 Transformare pentru Rețeaua Neuronală

1. **Centrare:** Alinierea imaginii în funcție de centrul de masă al pixelilor negri (`center_image_by_mass`).
2. **Redimensionare:** Aducerea la **160x160 px**.
3. **Conversie Cromatică:** Transformare din Grayscale/BGR în RGB.
4. **Normalizare:** Aplicarea funcției `preprocess_input` din MobileNetV2 (scalează pixelii în intervalul `[-1, 1]`).

### 4.3 Pregătirea pentru Antrenare (Training Split)

Datele din `data/train` sunt încărcate dinamic.

* **Encoding:** Etichetele (numele studenților) sunt transformate în vectori one-hot (`to_categorical`).
* **Augmentare:** (Implicită prin variația semnăturilor înrolate).

### 4.4 Salvarea rezultatelor preprocesării

* **Date Procesate:** Imaginile decupate și normalizate (160x160px) sunt salvate automat în `data/processed_sig/` (intermediar) și `data/train/` (organizat pe clase).
* **Model:** Modelul antrenat este serializat în fișierul `semnatura_model.h5`.
* **Configurație:** Parametrii de preprocesare sunt integrați direct în codul sursă `svas_web.py`.

## 5. Fișiere Generate în Această Etapă

* `data/raw_sig/` – PDF-urile brute încărcate pentru înrolare (sursa datelor de antrenare).
* `data/raw_list/` – PDF-urile brute (liste de prezență) folosite pentru testare/verificare.
* `data/processed_sig/` – Imagini intermediare decupate și normalizate (utilizate pentru debugging/verificare).
* `data/train/` – **Dataset-ul final**: imagini PNG (160x160px) organizate în foldere după numele studentului.
* `semnatura_model.h5` – Fișierul binar ce conține modelul neuronal antrenat (MobileNetV2 + Custom Head).
* `svas_web.py` – Scriptul principal (include logica de preprocesare, antrenare și interfața web).
* `requirements.txt` – Lista bibliotecilor necesare (`tensorflow`, `opencv`, `flask`, `pymupdf`).
## 6. Stare Etapă

* [x] Structură proiect definită și implementată (`svas_web.py`).
* [x] Pipeline de achiziție date din PDF funcțional.
* [x] Algoritm de segmentare și curățare (ROI extraction) implementat.
* [x] Preprocesare specifică CNN (Resize 160px, Normalizare).
* [x] Generare dataset structurat în folderul `data/train`.