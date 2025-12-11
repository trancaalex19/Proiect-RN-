Mai jos îți dau **tot blocul complet refăcut**, cu toate modificările de formatare aplicate, astfel încât să se afișeze corect în GitHub (tabele, structură repo, secțiuni, tot).
Este gata de copiat în **README_Etapa5.md** fără alte ajustări.

---

# **ETAPA 5 – Refacerea completă a fișierului (versiune finală, cu formatare corectă)**

## 1. Descrierea Caracteristicilor

### **Caracteristică: Imagine (X)**

| Proprietate        | Detalii                            |
| ------------------ | ---------------------------------- |
| **Tip**            | Matrice                            |
| **Unitate**        | Pixeli                             |
| **Descriere**      | Imaginea semnăturii redimensionată |
| **Domeniu valori** | 64 x 64 px                         |

### **Canal Culoare**

| Proprietate   | Detalii                 |
| ------------- | ----------------------- |
| **Tip**       | Numeric                 |
| **Unitate**   | –                       |
| **Descriere** | Intensitate (grayscale) |
| **Valoare**   | 1                       |

### **Intensitate Pixel**

| Proprietate        | Detalii                |
| ------------------ | ---------------------- |
| **Tip**            | Numeric                |
| **Descriere**      | Valoarea luminozității |
| **Domeniu valori** | 0 (Negru) – 255 (Alb)  |

---

## 2. Eticheta (Y)

| Proprietate         | Detalii                  |
| ------------------- | ------------------------ |
| **Tip**             | Categorial               |
| **Valori posibile** | 0 = falsă, 1 = autentică |
| **Descriere**       | Clasa asociată imaginii  |

---

## 3. Modelul Utilizat (CNN)

Arhitectura rețelei convoluționale:

```text
Input: 64x64x1 (grayscale)

[Conv2D 32 filtre, kernel 3x3]  
[ReLU]  
[MaxPooling 2x2]

[Conv2D 64 filtre, kernel 3x3]  
[ReLU]  
[MaxPooling 2x2]

[Conv2D 128 filtre, kernel 3x3]  
[ReLU]  
[MaxPooling 2x2]

[Flatten]

[Dense 128 neuroni]  
[ReLU]  
[Dropout 0.5]

Output: Dense (sigmoid, 1 neuron)
```

Funcția de pierdere: **binary_crossentropy**
Optimizer: **Adam**
Metrici: **accuracy**

---

## 4. Procedura de Antrenare

### **Set de date**

* 100 imagini totale

  * 50 semnături autentice
  * 50 semnături false

### **Preprocesare**

* Conversie în grayscale
* Redimensionare la 64×64 px
* Normalizare: `pixel / 255.0`
* Shuffle + split automat în antrenare/validare

---

## 5. Rezultate inferență și capturi de ecran

Exemplu rezultat într-o predicție reală (cu aplicația Flask):

**Imagine Reală:**
Se generează un scor al probabilității și se afișează:

```
Probabilitate autenticitate: 0.87
Clasificare: Semnătură Autentică
```

Screenshot-ul se află în:

```
docs/screenshots/inference_real.png
```

---

## 6. Codul complet al aplicației (svas_web.py)

```python
[ aici se inserează codul tău complet ]
```

(dacă vrei, pot să-l lipesc eu pentru tine)

---

## 7. Structura Repository-ului

Aceasta este versiunea care se afișează corect pe GitHub:

```text
SVAS-Project/
├── README.md                # Overview general
├── README_Etapa5.md         # Acest fișier
├── svas_web.py              # Aplicația completă (Modulele 1, 2, 3)
├── semnatura_model.h5       # Modelul antrenat și salvat
├── dataset/
│   ├── Date autentice/      # 50 imagini autentice
│   └── Date false/          # 50 imagini false
├── docs/
│   └── screenshots/
│       └── inference_real.png   # Screenshot cu predicția
└── requirements.txt
```

---

Dacă vrei, pot să-ți:

1. **Integrez codul complet în secțiunea 6**
2. **Generez automat întreg README.md final**
3. **Refac și Etapa 4 sau Etapa 6 dacă ai nevoie**

Vrei să includ și codul complet în document?
