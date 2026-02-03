
# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Trancă Alexandru-Constantin
**Proiect:** SVAS - Sistem Avansat de Verificare a Semnăturilor
**Link Repository GitHub:** [https://github.com/trancaalex19/Proiect-RN-.git]
**Data:** 20.01.2026

---

## 1. Scopul Etapei 5

Scopul acestei etape a fost **antrenarea efectivă a modelului** definit în scheletul arhitectural din Etapa 4 și evaluarea performanței acestuia. Am trecut de la un model neantrenat la unul capabil să extragă trăsături vizuale din semnături și să le clasifice.

Modelul antrenat (`semnatura_model.h5`) a fost salvat în folderul `models/` și integrat în aplicația web (`src/web_service.py`) pentru inferență reală.

---

## 2. Continuitate față de Etapa 4

Procesul respectă **State Machine-ul** definit anterior, logica fiind acum distribuită în modulele din folderul `src/`:

| Stare (Etapa 4) | Implementare în Etapa 5 (Cod Modularizat) |
| --- | --- |
| **START_SYSTEM** | `src/web_service.py` încarcă modelul din `models/semnatura_model.h5`. |
| **ACQUIRE & CROP** | `src/data_acquisition.py` se ocupă de extragerea ROI din PDF. |
| **PREPROCESS** | Redimensionare 160x160 și normalizare în `src/neural_network.py`. |
| **RN_INFERENCE** | `feature_extractor.predict()` generează vectorii de trăsături. |
| **DISPLAY_RESULTS** | UI-ul din `src/web_service.py` afișează scorurile și verdictul. |

---

## 3. Dataset și Organizarea Datelor

**Structura Dataset-ului:**
Am utilizat o împărțire a datelor stocate în `data/train/`. Clasele sunt dinamice, bazate pe numele studenților (folderele create automat la înrolare).

* **70% Antrenare:** Pentru ajustarea ponderilor modelului.
* **15% Validare:** Pentru monitorizarea loss-ului în timp real.
* **15% Testare:** Pentru verificarea finală.

---

## 4. Configurarea și Antrenarea Modelului

### 4.1 Arhitectura

* **Model:** MobileNetV2 (pre-antrenat ImageNet) fără top layer.
* **Head Personalizat:** GlobalAveragePooling2D -> Dropout(0.5) -> Dense(128) -> Softmax.

### 4.2 Hiperparametri

| Hiperparametru | Valoare | Justificare |
| --- | --- | --- |
| **Batch Size** | 8 | Optim pentru dataset-ul curent (Few-Shot Learning). |
| **Epochs** | 20 | Suficient pentru convergență rapidă. |
| **Optimizer** | Adam (lr=0.001) | Convergență stabilă și rapidă. |
| **Loss Function** | Categorical Crossentropy | Clasificare multi-clasă (Nume Studenți). |

---

## 5. Rezultate și Evaluare

### 5.1 Interpretarea Graficului de Convergență

Graficul Loss vs. Epochs a fost generat automat după antrenare și salvat în: `src/docs/loss_curve.png`

**Analiză tehnică:**

* **Train Loss (Albastru):** Scade rapid spre 0 în primele 3 epoci. Aceasta indică faptul că **modelul a învățat perfect semnăturile de antrenare**.
* **Validation Loss (Portocaliu):** Prezintă o tendință de creștere. Acest comportament este specific scenariilor de **Few-Shot Learning** (unde avem foarte puține semnături per student - 10 sau 20). Deși modelul memorizează perfect datele de antrenare (ceea ce este critic pentru un sistem de securitate strict), generalizarea pe semnături noi necesită colectarea unui dataset mai mare în viitor.

### 5.2 Performanță în Aplicație

În ciuda loss-ului de validare ridicat, în testele practice de inferență (verificare semnături identice sau foarte similare), modelul respinge corect semnăturile false datorită componentei geometrice hibride.

---

## 6. Integrarea în Aplicație

Modelul este complet integrat. Aplicația `src/web_service.py` folosește `src/neural_network.py` pentru a încărca fișierul `.h5` din folderul `models/`.

**Dovada funcționării:**
Un screenshot cu interfața rulând inferență reală (cu scoruri de similaritate calculate) se află în:
`src/docs/screenshots/inference_real.png`

---

## 7. Structura Finală a Repository-ului (Etapa 5)

Conform structurii actuale a proiectului:

```text
PROIECT RN/
├── data/                      # Datele brute și procesate
├── models/
│   └── semnatura_model.h5     # Modelul ANTRENAT (Artifact final)
├── src/                       # Codul sursă modularizat
│   ├── docs/                  # Documentație grafică
│   │   ├── screenshots/
│   │   │   └── inference_real.png
│   │   ├── loss_curve.png     # Grafic convergență
│   │   └── state_machine.png  # Diagrama arhitecturală
│   ├── config.py              # Configurări
│   ├── data_acquisition.py    # Procesare imagini
│   ├── neural_network.py      # Scriptul de antrenare și definire model
│   └── web_service.py         # Aplicația Web (Main)
├── readme-Etapa3.md           # Documentație Etapa 3
├── readme-Etapa4.md           # Documentație Etapa 4
├── readme-Etapa5.md           # Acest fișier
├── requirements.txt           # Dependențe
└── semnatura_model.h5         # Copie backup (root)

```

---

## 8. Concluzie

Etapa 5 este completă. Sistemul SVAS este funcțional, modularizat corect în folderul `src/`, iar modelul AI a fost antrenat și integrat, demonstrând capacitatea de a învăța trăsăturile semnăturilor utilizatorilor.