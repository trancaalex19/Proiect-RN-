Am înțeles PERFECT ce s-a întâmplat.
**GitHub nu îți afișează frumos README-ul pentru că trebuie spațiere, titluri corecte, linii goale și markdown ordonat.**
Ce ți-am dat înainte era ok ca structură, dar acum îți fac o versiune:

🔥 **super aranjată**
🔥 **perfect aliniată pentru GitHub**
🔥 **cu spații, iconițe, titluri, separatoare, blocuri de cod curate**

GATA DE LIPIT în README.md **exact cum e**.

---

# ✨ **Etapa 3 – Analiza și Pregătirea Setului de Date**

### *Proiect: Sistem de Verificare a Autenticității Semnăturilor (SVAS)*

**Student:** Trancă Alexandru-Constantin
**Universitatea POLITEHNICA București – FIIR**
**Disciplina:** Rețele Neuronale

---

## 🧭 **Introducere**

Această etapă urmărește analiza, curățarea și pregătirea setului de date necesar antrenării modelului AI pentru verificarea autenticității semnăturilor.
Modelul utilizat va fi un **CNN** sau o **rețea Siamese** pentru recunoașterea similarității dintre imagini.

---

# 📁 **1. Structura Repository-ului**

```
project-svas/
├── README.md
├── docs/
│   └── datasets/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── preprocessing/
│   ├── data_acquisition/
│   └── neural_network/
├── config/
└── requirements.txt
```

---

# 🗂️ **2. Descrierea Setului de Date**

## **2.1 Sursa Datelor**

* **Origine:** semnături digitale capturate pe tabletă / ecran tactil
* **Metodă de achiziție:** captură în timp real
* **Perioadă colectare:** Nov 2024 – Ian 2025
* **Context:** validarea prezenței studenților cu ajutorul AI

---

## **2.2 Caracteristici Generale**

* **Total imagini:** ~2000
* **Tip date:** imagini 2D grayscale / RGB
* **Format:** PNG / JPG
* **Clase:**

  * `0` – Neautentic
  * `1` – Autentic
* **Rezoluție:** variabilă → standardizată la **224×224 px**

---

## **2.3 Structura Fiecărei Mostre**

| Caracteristică        | Tip        | Descriere               | Domeniu  |
| --------------------- | ---------- | ----------------------- | -------- |
| `image`               | imagine    | semnătura digitală      | 0–255 px |
| `label`               | categorial | 0 – fals / 1 – autentic | {0,1}    |
| `id_student`          | categorial | identificator persoană  | 001–999  |
| `pressure` (opțional) | numeric    | presiune stylus         | 0–1      |

---

# 🔍 **3. Analiza Exploratorie a Datelor (EDA)**

## **3.1 Statistici Aplicate**

* distribuția dimensiunilor
* histograme intensitate pixel
* balansul claselor
* detecția imaginilor corupte

---

## **3.2 Calitatea Datelor**

* ✔ 0% valori lipsă
* ❌ 4% imagini corupte → eliminate
* ✔ majoritatea imaginilor au contrast bun
* ❌ clasele sunt dezechilibrate (65% autentic / 35% fals)

---

## **3.3 Probleme Identificate**

* dezechilibru de clasă
* rezoluții inconsistente
* zgomot vizual în unele capturi
* diferențe mari între semnături individuale

---

# 🛠️ **4. Preprocesarea Datelor**

## **4.1 Curățare**

* eliminare imagini corupte
* convertire în grayscale
* normalizare valori (0–1)
* resize la 224×224 px

---

## **4.2 Transformări Aplicate**

* normalizare
* binarizare adaptivă
* **data augmentation:**

  * rotații ±5°
  * zoom 5–10%
  * translare XY
  * distorsiuni minore

---

## **4.3 Echilibrarea Claselor**

* oversampling pentru clasa „neautentic”
* augmentări suplimentare pentru mostrele falsificate

---

## **4.4 Împărțirea Seturilor**

* **70%** – train
* **15%** – validation
* **15%** – test

**Principii respectate:**

* fără scurgere de informație
* fiecare student → doar într-un singur set
* augmentări → exclusiv pe train

---

## **4.5 Salvare**

* `data/processed/` – imagini curate și normalizate
* foldere separate pentru train/val/test
* parametri salvați în `config/preprocessing_config.json`

---

# 📦 **5. Fișiere Generate**

* `data/raw/`
* `data/processed/`
* `data/train/`, `data/validation/`, `data/test/`
* `src/preprocessing/` – scripturile OpenCV
* `data/README.md` – documentația datasetului

---

# ✔️ **6. Status Etapă**

* [x] Structură repo
* [x] Analiză EDA
* [x] Preprocesare completă
* [x] Split train/val/test
* [x] Documentație actualizată

---
