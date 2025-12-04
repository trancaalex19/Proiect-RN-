📘 README – Etapa 4

Disciplina: Rețele Neuronale

Instituție: POLITEHNICA București – FIIR

Student: Tranca Alexandru-Constantin

Data: 04.12.2025

Introducere

Acest document descrie activitățile realizate în Etapa 3, în care se analizează și se preprocesează setul de date necesar proiectului „Sistem de Verificare a Autenticității Semnăturilor (SVAS)”. Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

1. Structura Repository-ului Github (versiunea Etapei 3)

SVAS-Project/
├── README.md                # Documentația tehnică
├── docs/
│   └── datasets/            # Grafice și rapoarte distribuție
├── data/
│   ├── raw/                 # Date brute (imagini originale salvate din web app)
│   ├── processed/           # Date curățate (transformate intern în memorie)
│   ├── train/               # Set de instruire (gestionat automat)
│   ├── validation/          # Set de validare (split 20%)
│   └── test/                # Date de testare live
├── dataset/                 # Dataset-ul fizic
│   ├── Date autentice/      # 50 imagini originale (Clasa 1)
│   └── Date false/          # 50 imagini falsificate (Clasa 0)
├── src/
│   ├── preprocessing/       # Pipeline de redimensionare/normalizare
│   ├── data_acquisition/    # Modulul svas_web.py
│   └── neural_network/      # Arhitectura CNN
├── config/                  # Configurații (dimensiune 64x64)
└── requirements.txt         # tensorflow, flask, pillow, numpy


2. Descrierea Setului de Date

2.1 Sursa datelor

Origine: Date generate propriu (First-party data) prin aplicația svas_web.py.

Modul de achiziție: ☑ Senzori reali (Mouse / Touchpad) / ☐ Simulare / ☐ Fișier extern / ☐ Generare programatică.

Perioada / condițiile colectării: Noiembrie-Decembrie 2025. Colectare manuală prin desenare pe canvas digital HTML5.

2.2 Caracteristicile dataset-ului

Număr total de observații: 100 imagini.

Număr de caracteristici (features): 4096 (pixeli per imagine 64x64).

Tipuri de date: ☐ Numerice / ☐ Categoriale / ☐ Temporale / ☑ Imagini.

Format fișiere: PNG (Single Channel - Grayscale).

### 2.3 Descrierea fiecărei caracteristici

| Caracteristică   | Tip       | Unitate | Descriere                     | Domeniu valori          |
|-----------------|-----------|---------|-------------------------------|------------------------|
| Imagine (X)      | matrice   | pixeli  | Imaginea semnăturii redimensionată | 64 x 64 px             |
| Canal Culoare    | numeric   | -       | Intensitate (Grayscale)       | 1                      |
| Intensitate Pixel| numeric   | -       | Valoarea luminozității        | 0 (Negru) – 255 (Alb) |
| Etichetă (Y)     | categorial| -       | Clasa semnăturii             | {0: Fals, 1: Autentic} |


3. Analiza Exploratorie a Datelor (EDA) – Sintetic

3.1 Statistici descriptive aplicate

Distribuția Claselor: Dataset-ul este perfect echilibrat:

50 Semnături Autentice.

50 Semnături False.

Analiza Dimensiunilor: Toate imaginile sunt standardizate la 64x64 pixeli.

3.2 Analiza calității datelor

Detectarea valorilor lipsă: Nu există pixeli lipsă. Imaginile corupte (0 bytes) sunt ignorate automat.

Consistență: Formatul PNG lossless asigură calitatea liniilor desenate.

3.3 Probleme identificate

Variabilitate: Semnăturile cu mouse-ul prezintă un "tremur" specific (zgomot de cuantizare) față de cele pe hârtie.

Volum: Setul de 100 de date este mic, dar suficient pentru demonstrarea conceptului (Proof of Concept).

4. Preprocesarea Datelor

4.1 Curățarea datelor

Eliminare duplicatelor: Verificare manuală a folderelor.

Tratarea outlierilor: Eliminarea imaginilor complet albe (salvate eronat).

4.2 Transformarea caracteristicilor

Procesul este automatizat în codul Python:

Conversie Grayscale: Transformare RGB -> L (1 canal).

Redimensionare: Resize la 64x64 pixeli.

Normalizare: Împărțirea valorilor pixelilor la 255.0 => interval [0.0, 1.0].

4.3 Structurarea seturilor de date

Împărțire realizată:

80% – Train: Pentru învățarea ponderilor.

20% – Validation: Pentru monitorizarea performanței.

Principii respectate:

Shuffle: Amestecare aleatorie înainte de antrenare.

Stratificare: Asigurarea prezenței ambelor clase în validare.

4.4 Salvarea rezultatelor preprocesării

Datele nu sunt salvate intermediar pe disc, ci procesate "on-the-fly" în memoria RAM.

Modelul Final: Salvat ca semnatura_model.h5.

5. Diagrama Fluxului de Date

Mai jos este prezentat fluxul complet al datelor prin sistemul SVAS:

graph TD
    A[Utilizator] -->|Desenează Semnătura| B(Interfață Web - HTML Canvas)
    B -->|Apasă 'Verifică'| C{JavaScript}
    C -->|Codificare Base64| D[HTTP POST Request]
    D -->|Trimite datele| E[Server Python Flask]
    
    subgraph "Backend AI (Pre-procesare & Inferență)"
    E -->|Decodare Imagine| F[Imagine Brută]
    F -->|Resize 64x64 & Grayscale| G[Matrice 64x64x1]
    G -->|Normalizare /255.0| H[Tensor Input (0.0 - 1.0)]
    H -->|CNN Model| I[Rețea Neuronală]
    I -->|Predicție| J[Scor Sigmoid (0.0 - 1.0)]
    end
    
    J -->|Decizie (Prag > 0.8)| K[Verdict: AUTENTIC / FALS]
    K -->|Răspuns JSON| C
    C -->|Afișare Colorată| A


6. Fișiere Generate în Această Etapă

svas_web.py: Aplicația completă.

dataset/: Imaginile colectate.

semnatura_model.h5: Modelul antrenat.

README.md: Documentația.

7. Stare Etapă

[x] Structură repository configurată.

[x] Dataset analizat și echilibrat (50/50).

[x] Date preprocesate (Pipeline automat implementat).

[x] Seturi train/validation utilizate în antrenare.

[x] Aplicație Web funcțională și model antrenat.

[x] Documentație actualizată în README.
fa mi l sa arate mai frumos
