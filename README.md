✨ Etapa 3 – Analiza și Pregătirea Setului de Date

Proiect: Sistem de Verificare a Autenticității Semnăturilor (SVAS)

Student: Tranca Alexandru-Constantin
Grupa: 634 AB
Universitatea POLITEHNICA București – FIIR
Disciplina: Rețele Neuronale

🧭 1. Introducere

Această etapă a proiectului vizează colectarea, curățarea și preprocesarea datelor necesare pentru antrenarea rețelei neuronale.
Obiectivul principal a fost crearea unui dataset robust de semnături digitale și dezvoltarea unei interfețe web (svas_web.py) care integrează atât partea de achiziție de date, cât și cea de antrenare și inferență AI.

📁 2. Structura Repository-ului

Structura actualizată a proiectului la finalul Etapei 3:

SVAS-Project/
├── README.md                # Documentația curentă
├── svas_web.py              # Aplicația Web completă (Interfață + Backend AI)
├── semnatura_model.h5       # Modelul CNN antrenat și salvat
├── dataset/                 # Setul de date colectat
│   ├── Date autentice/      # 50 semnături originale (Clasa 1)
│   └── Date false/          # 50 semnături falsificate (Clasa 0)
└── requirements.txt         # Dependențe (tensorflow, flask, pillow, numpy)


🗂️ 3. Descrierea Setului de Date

3.1 Sursa Datelor

Origine: Date generate propriu (First-party data).

Metodă de achiziție: Desenare digitală folosind mouse/touchpad prin interfața aplicației web dezvoltate (svas_web.py).

Volum: Dataset inițial de 100 de imagini.

3.2 Distribuția Claselor

S-a urmărit un echilibru perfect al claselor pentru a evita bias-ul rețelei:

Clasă

Etichetă (Label)

Descriere

Număr Mostre

Autentic

1

Semnături realizate de titular

50

Fals

0

Încercări de imitare sau semnături aleatorii

50

🛠️ 4. Pipeline de Preprocesare

Înainte de a intra în Rețeaua Neuronală, imaginile brute trec printr-un proces automat de transformare implementat în Python:

Conversie Grayscale:

Transformare din RGB (3 canale) în L (1 canal).

Elimină informația inutilă de culoare, păstrând doar intensitatea liniilor.

Redimensionare (Resizing):

Toate imaginile sunt aduse la rezoluția standard de 64x64 pixeli.

Motiv: Reducerea complexității computaționale și standardizarea input-ului pentru CNN.

Normalizare:

Valorile pixelilor [0, 255] sunt împărțite la 255.0.

Rezultat: Valori float în intervalul [0.0, 1.0], esențiale pentru convergența rapidă a algoritmului Adam.

Data Augmentation (Implicit):

Variabilitatea naturală a desenului cu mouse-ul funcționează ca o augmentare a datelor, oferind diferențe subtile între mostre.

🧠 5. Arhitectura Modelului (Pe scurt)

Modelul utilizat pentru validarea datelor în această etapă este un CNN Secvențial:

Input: (64, 64, 1)

Feature Extraction: 2 straturi Conv2D + MaxPooling2D pentru detectarea trăsăturilor vizuale.

Clasificare: Strat Dense (128 neuroni) + Dropout (0.5 pentru evitare overfitting).

Output: Sigmoid (probabilitate 0-1).

💻 6. Aplicația Web (Livrabil Etapa 3)

S-a dezvoltat un serviciu web (svas_web.py) folosind Flask care permite:

✅ Desenarea semnăturilor direct în browser (HTML5 Canvas).

✅ Comunicarea asincronă cu backend-ul Python (Fetch API).

✅ Re-antrenarea modelului la cerere, folosind datele din folderul dataset/.

✅ Verificarea instantanee a semnăturilor noi.

✔️ 7. Status Etapă

[x] Colectare date: 50 Autentice / 50 False salvate în structura corectă.

[x] Curățare date: Eliminare imagini goale/corupte.

[x] Implementare Preprocesare: Resize și Normalizare integrate în cod.

[x] Dezvoltare Interfață: Aplicație Web funcțională.

[x] Validare: Modelul antrenat atinge o acuratețe preliminară satisfăcătoare (>90%).
e un readme in github
schimba mi niste cuvinte pe acolo
