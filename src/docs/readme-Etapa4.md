
# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Trancă Alexandru-Constantin
**Proiect:** SVAS - Sistem Avansat de Verificare a Semnăturilor
**Link Repository GitHub:** [https://github.com/trancaalex19/Proiect-RN-.git]
**Data:** 20.01.2026

---

## 1. Scopul Etapei 4

În această etapă am definit și implementat **arhitectura funcțională** a sistemului SVAS. Am livrat un schelet complet al aplicației software, unde modelul de Rețea Neuronală este definit, compilat și integrat într-un flux de lucru modular, capabil să ruleze end-to-end (de la încărcarea PDF-ului până la afișarea verdictului).

---

## 2. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| Nevoie (Reală) | Soluție SIA (Metrici) | Modul Software |
| --- | --- | --- |
| **Digitalizare Rapidă** | Extragere automată ROI din PDF → Dataset gata în < 10s | Modul 1 (Data Acquisition) |
| **Verificare Autenticitate** | Analiză Hibridă (MobileNetV2 + Geometrie) → Acuratețe > 90% | Modul 2 (Neural Network) |
| **Detectare Erori** | Analiză densitate pixeli → Alertă "NESEMNAT" instantanee | Modul 1 (Preprocessing) |

---

## 3. Contribuția Originală la Setul de Date

**Total observații finale:** ~200 imagini (estimat la finalul procesului de înrolare)
**Observații originale:** 200 (100%)

**Tipul contribuției:**

* [x] Date achiziționate cu senzori proprii (Scanare/Fotografiere documente reale)
* [x] Etichetare/adnotare manuală (Automatizată prin numele fișierului PDF)

**Descriere detaliată:**
Am creat formulare tipizate (liste de prezență și foi de semnături individuale) pe care le-am distribuit colegilor pentru a le semna. Documentele au fost digitalizate și procesate prin modulul propriu de achiziție (`data_acquisition.py`).
Acest proces asigură un set de date 100% original și relevant pentru mediul universitar. Etichetarea este automată: numele fișierului PDF (ex: `Popescu_Ion.pdf`) devine eticheta clasei (folderul `data/train/Popescu_Ion`).

**Locația codului:** `src/data_acquisition.py` (funcția `extract_signatures_blob_mode`)
**Locația datelor:** `data/raw_sig/` (PDF-uri sursă) și `data/train/` (dataset final)

---

## 4. Diagrama State Machine a Întregului Sistem

**Diagrama vizuală:** `docs/state_machine.png`

### Descrierea Fluxului (Text):

```text
IDLE → RECEIVE_WEB_REQUEST (Flask) → VALIDATE_FILE_TYPE (PDF Check) →
  │
  ├─ [Request: ENROLL] → CONVERT_PDF_TO_IMG → EXTRACT_ROIS (CV2 Contours) →
  │       PREPROCESS (Resize 160px, Binarize) → QUALITY_CHECK (min_pixels) →
  │         ├─ [Valid] → SAVE_TO_DATASET (folder structurat) → LOG_SUCCESS → IDLE
  │         └─ [Noise/Empty] → DISCARD_BLOB → LOG_INFO → IDLE
  │
  └─ [Request: VERIFY] → CONVERT_LIST_TO_IMG → EXTRACT_CANDIDATES →
          [LOOP: For each signature in list] → LOAD_REFERENCE_EMBEDDINGS →
          RN_INFERENCE (MobileNetV2) + GEOMETRIC_ANALYSIS (Density/Proj) →
          CALCULATE_FUSION_SCORE (Weighted Avg) →
            ├─ [Score ≥ 0.90] → SET_STATUS_AUTHENTIC → APPEND_TO_REPORT
            ├─ [Score < 0.90] → SET_STATUS_SUSPECT → APPEND_TO_REPORT
            └─ [No ROI] → SET_STATUS_NESEMNAT → APPEND_TO_REPORT
          [END LOOP] → RETURN_JSON_RESPONSE → UPDATE_UI → IDLE

      ↓ [Server Error / Bad File]
    HANDLE_EXCEPTION → RETURN_ERROR_MESSAGE → IDLE

```

### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip **Event-Driven (Web Request / Batch Processing)** deoarece aplicația SVAS nu funcționează ca un sistem de monitorizare continuă, ci răspunde punctual la cererile utilizatorului.

Fluxul este divizat în două ramuri critice:

1. **Înrolare (Build DB):** Procesează batch-uri de semnături pentru antrenare.
2. **Verificare (Inference):** Compară semnăturile dintr-o listă nouă cu baza de date existentă.

Această separare asigură robustețea sistemului și tratarea corectă a erorilor (fișiere corupte, pagini goale) fără a bloca serverul.

---

## 5. Scheletul Complet al celor 3 Module

Sistemul este implementat modular, respectând cerințele de decuplare a logicii:

| Modul | Fișier Sursă (Python) | Funcționalități Implementate (Status: OK) |
| --- | --- | --- |
| **1. Data Acquisition**<br>

<br>*(Înrolare)* | `src/data_acquisition.py` | • Conversie PDF → Imagine HD (fitz)<br>

<br>• Segmentare automată (ROI Extraction)<br>

<br>• Preprocesare geometrică (Centrare, Proiecții) |
| **2. Neural Network**<br>

<br>*(Model AI)* | `src/neural_network.py` | • Definire MobileNetV2 + Custom Dense Head<br>

<br>• Salvare/Încărcare model (`semnatura_model.h5`)<br>

<br>• Calcul similaritate vectori (Inference) |
| **3. Web Service / UI**<br>

<br>*(Interfață)* | `src/web_service.py` | • Server Flask (Port 5000)<br>

<br>• Interfață HTML5/JS (Single Page App)<br>

<br>• Rutare cereri și afișare rezultate (Verde/Roșu) |

---

## 6. Structura Repository-ului la Finalul Etapei 4

```text
proiect-rn-tranca-alexandru/
├── data/
│   ├── raw_sig/           # [Input] PDF-uri brute pentru înrolare
│   ├── raw_list/          # [Input] PDF-uri brute pentru verificare
│   ├── processed_sig/     # [Intermediar] Semnături decupate
│   ├── processed_list/    # [Intermediar] Verificare vizuală
│   └── train/             # [Dataset Final] Structurat pe foldere (Clase)
├── docs/
│   ├── state_machine.png  # Diagrama fluxului logic
│   └── screenshots/       
│       └── ui_demo.png    # Captură interfață funcțională
├── src/
│   ├── config.py          # Configurări globale (căi, parametri)
│   ├── data_acquisition.py # MODUL 1: Prelucrare imagini
│   ├── neural_network.py   # MODUL 2: Logică AI (TensorFlow)
│   └── web_service.py      # MODUL 3: Entry point (Aplicația principală)
├── semnatura_model.h5     # Modelul RN definit și compilat
├── README.md              # Documentație generală
├── README_Etapa3.md       # Documentație Etapa 3
├── README_Etapa4_Arhitectura_SIA.md  # ← Acest fișier
└── requirements.txt       # Dependențe proiect

```

---

## 7. Checklist Final

### Documentație și Structură

* [x] Tabelul Nevoie → Soluție completat.
* [x] Declarație contribuție date originale (100% scanări proprii).
* [x] Cod modularizat în folderul `src/`.
* [x] Diagrama State Machine salvată în `docs/`.
* [x] Screenshot UI salvat în `docs/screenshots/`.

### Funcționalitate Module

* [x] **Modul 1:** `data_acquisition.py` procesează corect PDF-urile și extrage ROI-urile.
* [x] **Modul 2:** `neural_network.py` gestionează modelul `.h5` și calculează scorurile.
* [x] **Modul 3:** `web_service.py` pornește serverul și interfața răspunde la comenzi.

---

**Instrucțiuni de rulare:**

1. Instalați dependențele: `pip install -r requirements.txt`
2. Navigați în folderul sursă: `cd src`
3. Porniți aplicația: `python web_service.py`
4. Accesați în browser: `http://127.0.0.1:5000`