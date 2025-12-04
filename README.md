📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

Proiect: Sistem de Verificare a Autenticității Semnăturilor (SVAS)

Student: Tranca Alexandru-Constantin

Grupa: 634 AB

Instituție: Universitatea POLITEHNICA București – FIIR

Disciplina: Rețele Neuronale

🧭 1. Introducere

Această etapă a proiectului vizează colectarea, curățarea și preprocesarea datelor necesare pentru antrenarea rețelei neuronale.
Obiectivul principal a fost constituirea unui dataset robust de semnături digitale și dezvoltarea unei interfețe web (svas_web.py) care integrează funcționalitățile de achiziție de date, antrenare a modelului și inferență AI.

📁 2. Structura Repository-ului

Arhitectura proiectului la finalul Etapei 3:

SVAS-Project/
├── README.md                # Documentația tehnică a proiectului
├── svas_web.py              # Aplicația Web (Interfață Grafică + Backend AI)
├── semnatura_model.h5       # Modelul CNN antrenat și serializat
├── dataset/                 # Setul de date colectat
│   ├── Date autentice/      # 50 imagini cu semnături originale (Clasa 1)
│   └── Date false/          # 50 imagini cu semnături falsificate (Clasa 0)
└── requirements.txt         # Dependențe: tensorflow, flask, pillow, numpy



🗂️ 3. Descrierea Setului de Date

3.1 Sursa Datelor

Origine: Date generate propriu (First-party data).

Metodă de achiziție: Desenare digitală utilizând mouse-ul sau touchpad-ul, prin intermediul interfeței aplicației web dezvoltate (svas_web.py).

Volum: Dataset inițial compus din 100 de imagini.

3.2 Distribuția Claselor

S-a menținut un echilibru perfect al claselor pentru a preveni bias-ul rețelei neuronale în procesul de învățare:

Clasă

Etichetă (Label)

Descriere

Număr Mostre

Autentic

1

Semnături realizate de titularul contului

50

Fals

0

Încercări de imitare sau semnături aleatorii

50

3.3 Descrierea fiecărei caracteristici

Fiecare punct de date reprezintă o imagine procesată, definită prin următorii parametri:

Caracteristică

Tip

Unitate

Descriere

Domeniu valori

Imagine (Input)

Matrice

Pixeli

Reprezentarea vizuală a semnăturii (64x64)

0–255 (intensitate)

Canale Culoare

Numeric

-

Numărul de canale de culoare (Grayscale)

1

Valoare Pixel

Numeric

-

Valoarea normalizată a luminozității

0.0 – 1.0 (float)

Etichetă (Target)

Categorial

-

Clasa de apartenență (Autentic/Fals)

{0, 1}

3.4 Probleme Identificate

Variabilitate de Captură: Semnăturile realizate cu mouse-ul prezintă un zgomot specific ("tremur") comparativ cu cele olografe. Modelul a fost configurat să generalizeze peste aceste imperfecțiuni.

Dimensiune Dataset: Volumul de 100 de imagini este minimal pentru Deep Learning, însă suficient pentru validarea conceptului (Proof of Concept) în această etapă.

🛠️ 4. Pipeline de Preprocesare

Înainte de a fi introduse în Rețeaua Neuronală, imaginile brute parcurg un flux automat de transformare implementat în Python:

Conversie Grayscale:

Transformarea imaginii din spectrul RGB (3 canale) în L (1 canal).

Scop: Eliminarea redundanței cromatice și păstrarea doar a informației structurale (intensitatea liniilor).

Redimensionare (Resizing):

Standardizarea tuturor imaginilor la rezoluția de 64x64 pixeli.

Scop: Reducerea complexității computaționale și uniformizarea input-ului pentru CNN.

Normalizare:

Împărțirea valorilor pixelilor [0, 255] la 255.0.

Rezultat: Valori float în intervalul [0.0, 1.0], esențiale pentru convergența rapidă a algoritmului de optimizare (Adam).

Data Augmentation (Implicit):

Variabilitatea naturală indusă de desenarea manuală cu mouse-ul funcționează ca o augmentare a datelor, oferind diferențe subtile între mostrele de antrenament.

🧠 5. Arhitectura Modelului (Rezumat)

Modelul utilizat pentru validarea datelor în această etapă este un CNN Secvențial (Convolutional Neural Network):

Input Layer: (64, 64, 1)

Feature Extraction: 2 straturi de tip Conv2D urmate de MaxPooling2D pentru detectarea trăsăturilor vizuale locale.

Classification Head: Strat Dense (128 neuroni) + Dropout (0.5 pentru prevenirea overfitting-ului).

Output Layer: Funcție de activare Sigmoid (probabilitate 0-1).

💻 6. Aplicația Web (Livrabil Etapa 3)

S-a dezvoltat un serviciu web (svas_web.py) utilizând framework-ul Flask, care oferă următoarele funcționalități:

✅ Interfață de Captură: Desenarea semnăturilor direct în browser folosind HTML5 Canvas.

✅ Comunicare Asincronă: Transmiterea datelor către backend-ul Python prin Fetch API.

✅ Modul de Antrenare: Posibilitatea de a re-antrena modelul la cerere, utilizând datele stocate în folderul dataset/.

✅ Inferență în Timp Real: Verificarea instantanee a semnăturilor noi și afișarea verdictului.

📦 7. Fișiere Generate în Această Etapă

svas_web.py: Codul sursă complet al aplicației (Server Web + Logică AI).

dataset/: Directorul conținând cele 100 de imagini colectate și clasificate.

semnatura_model.h5: Fișierul binar al modelului antrenat, gata de utilizare.

README.md: Documentația tehnică actualizată a proiectului.

✔️ 8. Status Etapă

$$x$$

 Colectare date: 50 mostre Autentice / 50 mostre False salvate și structurate corect.

$$x$$

 Curățare date: Eliminarea imaginilor goale sau corupte.

$$x$$

 Implementare Preprocesare: Integrarea funcțiilor de Resize și Normalizare în cod.

$$x$$

 Dezvoltare Interfață: Aplicație Web funcțională și testată.

$$x$$

 Validare: Modelul antrenat a atins o acuratețe preliminară satisfăcătoare (>90%).
Nu imi schimba nimic din el doar fa mi l sa arate bine ca sa il pun intru un fisier readme de pe github
