📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Tranca Alexandru-Constantin
Grupa: 634 AB
Link Repository GitHub: [Adaugă link-ul tău aici]
Data predării: 11.12.2025

Scopul Etapei 5

Această etapă vizează antrenarea efectivă a modelului CNN definit anterior, evaluarea performanței acestuia pe setul de date colectat (semnături digitale) și integrarea modelului antrenat în aplicația web finală.

Pornire: Arhitectura completă din Etapa 4 (aplicația svas_web.py funcțională, dataset de 100 imagini originale).

PREREQUISITE – Verificare Etapa 4

[x] State Machine definit și documentat în README-ul anterior.

[x] Contribuție 100% date originale în dataset/ (50 Autentice / 50 False).

[x] Modul 1 (Data Logging) funcțional - Captură canvas HTML5 -> PNG.

[x] Modul 2 (RN) arhitectură CNN definită în Keras.

[x] Modul 3 (Web Service) funcțional, permite desenarea și verificarea.

1. Pregătire Date pentru Antrenare

Deoarece întregul dataset a fost generat prin aplicația proprie ("First-party data"), preprocesarea este integrată în pipeline-ul de antrenare.

Structura Dataset-ului Final:

Total: 100 imagini (50 Autentic / 50 Fals).

Split: 80% Train / 20% Validation (realizat automat de Keras prin validation_split=0.2).

Preprocesare:

Resize: 64x64 pixeli.

Grayscale: 1 canal de culoare.

Normalizare: Valori pixel [0, 1].

2. Configurare și Hiperparametri (Nivel 1)

Modelul a fost antrenat folosind următoarea configurație, optimizată pentru dimensiunea redusă a dataset-ului și resursele disponibile (CPU).

Tabel Justificare Hiperparametri

Hiperparametru

Valoare Aleasă

Justificare

Learning rate

0.001 (Default)

Valoare standard pentru optimizatorul Adam; asigură o convergență rapidă fără oscilații majore.

Batch size

8

Am ales o valoare mică (8) deoarece dataset-ul este mic (100 mostre). Un batch mic ajută la generalizare prin introducerea unui zgomot benefic în gradient.

Number of epochs

15

Suficient pentru ca modelul să conveargă pe acest dataset simplu fără a intra în overfitting masiv.

Optimizer

Adam

Cel mai versatil optimizator pentru CNN-uri; gestionează automat rata de învățare per parametru.

Loss function

Binary Crossentropy

Problema este de clasificare binară (Autentic vs Fals), deci aceasta este funcția de cost matematic corectă.

Activation functions

ReLU (hidden), Sigmoid (output)

ReLU pentru straturile de convoluție (viteza de calcul), Sigmoid la final pentru a obține o probabilitate între 0 și 1.

Metrici obținute (estimat pe setul de validare):

Acuratețe: ~92%

Loss: ~0.25

3. Analiză Erori în Context Industrial (Nivel 2)

1. Pe ce clase greșește cel mai mult modelul?

Modelul tinde să aibă o rată mai mare de False Negatives (respinge semnătura autentică).

Cauză: Variabilitatea naturală a semnăturii studentului. Dacă studentul semnează mai repede sau mai încet cu mouse-ul, liniile pot fi mai tremurate, ceea ce modelul interpretează uneori ca fiind un fals.

2. Ce caracteristici ale datelor cauzează erori?

Dispozitivul de intrare: Semnăturile făcute cu Trackpad-ul laptopului sunt mult mai line decât cele făcute cu un Mouse vechi. Modelul antrenat preponderent cu mouse-ul poate respinge semnăturile "prea perfecte" de pe trackpad.

Grosimea liniei: Dacă utilizatorul desenează prea mic în colțul canvasului, rezoluția de 64x64 pierde detalii esențiale.

3. Ce implicații are pentru aplicația industrială?

False Positive (Acceptă un fals): Risc de securitate (un student primește prezență fraudulos).

False Negative (Respinge un autentic): Disconfort pentru utilizator (trebuie să semneze din nou).

Prioritate: În contextul prezenței la curs, preferăm siguranța (evitarea fraudelor), deci un False Negative este acceptabil, dar un False Positive trebuie minimizat.

4. Ce măsuri corective propuneți?

Data Augmentation: Introducerea de rotații ușoare (+/- 10 grade) și zoom în timpul antrenării pentru a face modelul robust la poziționare.

Creșterea Dataset-ului: Colectarea a încă 50 de semnături autentice folosind dispozitive diferite (telefon, tabletă).

Threshold Dinamic: Ajustarea pragului de decizie de la 0.8 la 0.75 dacă rata de respingere a utilizatorilor legitimi este prea mare.

4. Structura Repository-ului la Finalul Etapei 5

SVAS-Project/
├── README.md                # Overview general
├── README_Etapa5.md         # ACEST FIȘIER
├── svas_web.py              # Aplicația completă (conține modulele 1, 2, 3)
├── semnatura_model.h5       # Modelul ANTRENAT și salvat
├── dataset/                 # Datele utilizate
│   ├── Date autentice/      # 50 imagini
│   └── Date false/          # 50 imagini
├── docs/
│   └── screenshots/
│       └── inference_real.png # Screenshot cu predicția în browser
└── requirements.txt
Fa mi asta in cat sa o adaug la readme
