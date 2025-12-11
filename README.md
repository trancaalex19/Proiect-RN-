
# **ETAPA 5 – INTEGRAREA COMPONENTELOR ȘI REALIZAREA APLICAȚIEI FINALE**

## **1. Obiectivul Etapei 5**

Scopul acestei etape este integrarea tuturor modulelor dezvoltate anterior (preprocesare, clasificare, antrenare model) într-o aplicație funcțională ce permite:

* încărcarea unei imagini cu o semnătură,
* preprocesarea automată a imaginii,
* inferența utilizând modelul antrenat,
* afișarea rezultatului direct în browser.

Această etapă finalizează fluxul complet al proiectului.

---

## **2. Arhitectura Sistemului Integrat**

În această etapă, aplicația finală este structurată în **3 module funcționale**, care rulează împreună în `svas_web.py`:

### **2.1 Modulul 1 – Preprocesare Imagine**

Include:

* conversie la grayscale,
* redimensionare la 64×64 px,
* normalizare valori de intensitate,
* transformare în tensor utilizabil de model.

### **2.2 Modulul 2 – Modelul de Clasificare**

* Model CNN antrenat anterior (fișierul `semnatura_model.h5`).
* Importat și încărcat automat la pornirea aplicației.

### **2.3 Modulul 3 – Interfața Web (Flask)**

Permite:

* upload imagine,
* rularea pipeline-ului de inferență,
* afișarea rezultatului: Autentică / Falsă.

---

## **3. Caracteristicile și Structura Datelor**

### **3.1 Descriere generală**

Sistemul funcționează pe baza unor imagini grayscale 64×64 px, utilizate ca intrare în modelul CNN.

### **3.2 Tabelul caracteristicilor**

```markdown
| Caracteristică      | Tip        | Unitate | Descriere                          | Domeniu valori            |
|---------------------|-----------|---------|------------------------------------|----------------------------|
| Imagine (X)          | matrice   | pixeli  | Imaginea semnăturii procesate      | 64 × 64 px                 |
| Canal culoare        | numeric   | -       | Canal grayscale                     | 1                          |
| Intensitate pixeli   | numeric   | -       | Valoarea luminanței                 | 0 (negru) – 255 (alb)      |
| Etichetă (Y)         | categorial| -       | Clasa semnăturii                    | {0: Falsă, 1: Autentică}   |
```

---

## **4. Structura Repository-ului la Finalul Etapei 5**

```text
SVAS-Project/
│
├── README.md                    # Overview general
├── README_Etapa5.md             # ACESȚI FIȘIER
│
├── svas_web.py                  # Aplicația finală (Module 1, 2, 3 integrate)
│
├── semnatura_model.h5           # Modelul CNN antrenat
│
├── dataset/
│   ├── Date autentice/          # 50 imagini
│   └── Date false/              # 50 imagini
│
├── docs/
│   └── screenshots/
│       └── inference_real.png   # Screenshot cu predicția în browser
│
└── requirements.txt             # Dependințe Python
```

---

## **5. Funcționarea Aplicației Web**

Aplicația este implementată în **Flask** și realizează:

1. **Încărcarea fișierului** prin formular HTML.
2. **Validarea tipului de fișier.**
3. **Preprocesarea imaginii** conform pipeline-ului stabilit.
4. **Rularea predicției** utilizând modelul `semnatura_model.h5`.
5. **Afișarea rezultatului** împreună cu imaginea încărcată.

---

## **6. Rularea aplicației**

### **6.1 Instalarea dependențelor**

```bash
pip install -r requirements.txt
```

### **6.2 Pornirea aplicației**

```bash
python svas_web.py
```

Aplicația devine accesibilă pe:

```
http://127.0.0.1:5000/
```

---

## **7. Exemplu de Inferență (Captură ecran)**

Imaginea de mai jos este inclusă în repository:

```
docs/screenshots/inference_real.png
```

Aceasta demonstrează afișarea predicției în browser.

---

## **8. Concluzia Etapei 5**

Această etapă marchează finalizarea proiectului prin:

* integrarea tuturor modulelor,
* dezvoltarea unei interfețe utilizabile,
* rularea completă a fluxului input–preprocesare–predicție–afișare rezultat.

Sistemul este acum complet funcțional și poate fi extins pentru dataset-uri mai mari, modele mai robuste sau UI îmbunătățită.

---
