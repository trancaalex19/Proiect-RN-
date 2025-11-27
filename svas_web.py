import os
import numpy as np
import threading
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import io
import base64
import sys
import subprocess

# --- DIAGNOSTIC & IMPORT TENSORFLOW ---
print("\n" + "="*60)
print(f"[DIAGNOSTIC] Python rulează din: {sys.executable}")
print("Încerc să import TensorFlow...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    TF_AVAILABLE = True
    print("[OK] TensorFlow detectat cu succes!")
    print(f"[OK] Versiune: {tf.__version__}")
except ImportError as e:
    TF_AVAILABLE = False
    print("\n[EROARE] TensorFlow NU a fost găsit.")
    print("-" * 60)
    print("SOLUȚIE: Copiază și rulează comanda de mai jos în terminal:")
    # Generăm comanda exactă pentru python-ul curent
    print(f'"{sys.executable}" -m pip install tensorflow')
    print("-" * 60)
except Exception as e:
    TF_AVAILABLE = False
    print(f"[EROARE NECUNOSCUTĂ]: {e}")

print("="*60 + "\n")

# --- CONFIGURĂRI ---
app = Flask(__name__)

# Configurăm căile exact cum ai spus tu
DATASET_DIR = "dataset"
FOLDER_REAL = "Date autentice"
FOLDER_FAKE = "Date false"
MODEL_PATH = "semnatura_model.h5"

# --- INTERFAȚA WEB (HTML + CSS + JS) ---
# Includem HTML-ul direct aici pentru a avea un singur fișier
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVAS - Verificare Web</title>
    <style>
        :root { --bg: #1a1a2e; --surface: #16213e; --primary: #0f3460; --accent: #e94560; --text: #eaeaea; }
        body { font-family: 'Segoe UI', sans-serif; background-color: var(--bg); color: var(--text); margin: 0; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
        h1 { margin-top: 20px; font-weight: 300; letter-spacing: 2px; }
        .container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin-top: 20px; }
        .card { background-color: var(--surface); padding: 20px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.3); }
        canvas { background-color: white; border-radius: 8px; cursor: crosshair; touch-action: none; }
        .controls { display: flex; flex-direction: column; gap: 10px; min-width: 250px; }
        button { padding: 12px; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s; color: white; }
        .btn-verify { background-color: #2CC985; } .btn-verify:hover { background-color: #229965; }
        .btn-clear { background-color: #555; } .btn-clear:hover { background-color: #777; }
        .btn-train { background-color: var(--primary); margin-top: 20px; } .btn-train:hover { background-color: #1a5c8e; }
        #result { margin-top: 20px; font-size: 24px; font-weight: bold; text-align: center; height: 40px; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid var(--accent); border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; display: none; margin: 0 auto;}
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .stats { font-size: 12px; color: #888; margin-top: 10px; text-align: center; }
        .error-banner { background-color: #e94560; color: white; padding: 10px; border-radius: 8px; margin-bottom: 10px; display: none; text-align: center;}
    </style>
</head>
<body>

    <h1>SVAS WEB INTERFACE</h1>

    <div class="container">
        <!-- ZONA DE CONTROL -->
        <div class="card controls">
            <h3>Panou Control</h3>
            <div id="tf-error" class="error-banner">ATENȚIE: TensorFlow nu este instalat pe server!</div>
            <p style="font-size: 14px; color: #ccc;">Desenează semnătura în dreapta și apasă Verifică.</p>
            
            <button class="btn-verify" onclick="verifySignature()">VERIFICĂ SEMNĂTURA</button>
            <button class="btn-clear" onclick="clearCanvas()">Șterge Tabla</button>
            
            <div id="result">Așteptare...</div>
            <div class="loader" id="loader"></div>

            <hr style="border-color: #333; width: 100%; margin: 15px 0;">
            
            <p style="font-size: 12px;">Admin Zone</p>
            <button class="btn-train" onclick="trainModel()">Re-Antrenează Modelul</button>
            <div class="stats" id="train-stats">Model status: Încărcat</div>
        </div>

        <!-- ZONA DE DESEN -->
        <div class="card">
            <canvas id="sigCanvas" width="500" height="300"></canvas>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('sigCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;

        // Verificăm statusul la încărcare
        window.onload = function() {
            fetch('/status').then(res => res.json()).then(data => {
                if(!data.tf_ok) {
                    document.getElementById('tf-error').style.display = "block";
                    document.getElementById('train-stats').innerText = "Model status: EROARE TF";
                }
            });
        };

        // Configurare stil desen
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';

        // Evenimente Mouse & Touch
        function start(e) { isDrawing = true; draw(e); }
        function end() { isDrawing = false; ctx.beginPath(); }
        function draw(e) {
            if (!isDrawing) return;
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const y = (e.clientY || e.touches[0].clientY) - rect.top;
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        canvas.addEventListener('mousedown', start);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', end);
        canvas.addEventListener('mouseout', end);
        canvas.addEventListener('touchstart', start);
        canvas.addEventListener('touchmove', draw);
        canvas.addEventListener('touchend', end);

        function clearCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').innerText = "Așteptare...";
            document.getElementById('result').style.color = "#eaeaea";
        }

        function verifySignature() {
            // Verificam daca e gol
            const blank = document.createElement('canvas');
            blank.width = canvas.width; blank.height = canvas.height;
            if(canvas.toDataURL() === blank.toDataURL()) {
                alert("Te rog să semnezi întâi!");
                return;
            }

            document.getElementById('result').innerText = "";
            document.getElementById('loader').style.display = "block";

            // Convertim desenul in imagine Base64
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = canvas.width; tempCanvas.height = canvas.height;
            const tCtx = tempCanvas.getContext('2d');
            tCtx.fillStyle = "white";
            tCtx.fillRect(0,0, tempCanvas.width, tempCanvas.height);
            tCtx.drawImage(canvas, 0, 0);
            
            const dataURL = tempCanvas.toDataURL('image/png');

            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loader').style.display = "none";
                const resDiv = document.getElementById('result');
                if(data.error) {
                    resDiv.innerText = "Eroare: " + data.error;
                    resDiv.style.color = "red";
                    alert("Eroare Server: " + data.error);
                } else {
                    resDiv.innerText = data.verdict + " (" + (data.score * 100).toFixed(1) + "%)";
                    resDiv.style.color = data.score > 0.8 ? "#2CC985" : "#e94560";
                }
            })
            .catch(err => {
                console.error(err);
                document.getElementById('loader').style.display = "none";
                alert("Eroare de comunicare cu serverul.");
            });
        }

        function trainModel() {
            if(!confirm("Ești sigur că vrei să re-antrenezi modelul folosind datele din foldere?")) return;
            
            document.getElementById('train-stats').innerText = "Se antrenează... Așteaptă...";
            fetch('/train', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
                document.getElementById('train-stats').innerText = data.message;
            });
        }
    </script>
</body>
</html>
"""

# --- BACKEND FLASK ---

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    return jsonify({"tf_ok": TF_AVAILABLE})

@app.route('/predict', methods=['POST'])
def predict():
    if not TF_AVAILABLE:
        return jsonify({"error": "TensorFlow lipsă! Verifică consola serverului."}), 500

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Modelul nu există. Antrenează-l întâi!"}), 400

    try:
        # 1. Primim imaginea Base64 de la JS
        data = request.json['image']
        # Eliminăm headerul "data:image/png;base64,"
        header, encoded = data.split(",", 1)
        binary_data = base64.b64decode(encoded)

        # 2. Procesăm imaginea (exact ca la antrenare)
        img = Image.open(io.BytesIO(binary_data)).convert('L') # Grayscale
        img = img.resize((64, 64)) # Resize la 64x64
        img_arr = np.array(img) / 255.0 # Normalizare
        img_arr = img_arr.reshape(1, 64, 64, 1)

        # 3. Predicție
        model = load_model(MODEL_PATH)
        prediction = model.predict(img_arr)
        score = float(prediction[0][0])

        verdict = "AUTENTIC" if score > 0.8 else "FALS"
        
        return jsonify({"verdict": verdict, "score": score})

    except Exception as e:
        print(f"Eroare predicție: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    if not TF_AVAILABLE:
        return jsonify({"message": "Eroare: TensorFlow lipsă. Instalează folosind comanda din consolă!"}), 500

    # Pornim antrenarea într-un thread separat ca să nu blocheze site-ul
    thread = threading.Thread(target=run_training_process)
    thread.start()
    return jsonify({"message": "Antrenarea a început în fundal! Urmărește terminalul."})

def run_training_process():
    print("--- ÎNCEPE RE-ANTRENAREA ---")
    data, labels = [], []

    # Încărcăm date AUTENTICE
    path_real = os.path.join(DATASET_DIR, FOLDER_REAL)
    if os.path.exists(path_real):
        for f in os.listdir(path_real):
            try:
                img = Image.open(os.path.join(path_real, f)).convert('L')
                arr = np.array(img) / 255.0
                data.append(arr.reshape(64,64,1))
                labels.append(1)
            except: pass
    
    # Încărcăm date FALSE
    path_fake = os.path.join(DATASET_DIR, FOLDER_FAKE)
    if os.path.exists(path_fake):
        for f in os.listdir(path_fake):
            try:
                img = Image.open(os.path.join(path_fake, f)).convert('L')
                arr = np.array(img) / 255.0
                data.append(arr.reshape(64,64,1))
                labels.append(0)
            except: pass

    if len(data) < 10:
        print("Prea puține date pentru antrenare!")
        return

    X = np.array(data)
    y = np.array(labels)

    # Re-creăm modelul
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X, y, epochs=15, batch_size=8, verbose=1)
    model.save(MODEL_PATH)
    print("--- MODEL RE-ANTRENAT ȘI SALVAT ---")

if __name__ == '__main__':
    # Rulăm serverul pe portul 5000
    app.run(debug=True, port=5000)