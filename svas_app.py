import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw
import os
import numpy as np
import random
import threading

# Încercăm să importăm TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- CONFIGURĂRI ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SignatureApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SVAS - Sistem Verificare Semnături (Neural Network)")
        self.geometry("1000x650")
        
        # Variabile
        self.brush_width = 4
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        self.image_size = (64, 64)
        self.model_path = "semnatura_model.h5"
        self.dataset_path = "dataset"
        self.TARGET_SAMPLES = 50  # Tinta setată

        # Creare foldere dataset
        os.makedirs(os.path.join(self.dataset_path, "autentic"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, "fals"), exist_ok=True)

        # --- INTERFAȚĂ ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # PANOU STÂNGA
        self.left_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nswe")

        self.logo_label = ctk.CTkLabel(self.left_frame, text="SVAS Pro", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        ctk.CTkLabel(self.left_frame, text="Panou Control", font=ctk.CTkFont(size=14)).grid(row=1, column=0, padx=20, pady=(0, 20))

        # 1. Colectare
        self.lbl_collect = ctk.CTkLabel(self.left_frame, text="1. Colectare Date (Target: 50)", text_color="gray", font=ctk.CTkFont(weight="bold"))
        self.lbl_collect.grid(row=2, column=0, sticky="w", padx=20)

        self.btn_save_real = ctk.CTkButton(self.left_frame, text="Salvează AUTENTIC (Tu)", fg_color="green", command=lambda: self.save_image("autentic"))
        self.btn_save_real.grid(row=3, column=0, padx=20, pady=5)

        self.btn_save_fake = ctk.CTkButton(self.left_frame, text="Salvează FALS (Altcineva)", fg_color="red", hover_color="darkred", command=lambda: self.save_image("fals"))
        self.btn_save_fake.grid(row=4, column=0, padx=20, pady=5)

        self.lbl_count = ctk.CTkLabel(self.left_frame, text=f"Statistici: 0/{self.TARGET_SAMPLES} | 0/{self.TARGET_SAMPLES}")
        self.lbl_count.grid(row=5, column=0, padx=20, pady=(5, 20))

        # 2. Antrenare
        self.lbl_train = ctk.CTkLabel(self.left_frame, text="2. Creierul (AI)", text_color="gray")
        self.lbl_train.grid(row=6, column=0, sticky="w", padx=20)

        self.btn_train = ctk.CTkButton(self.left_frame, text="Antrenează Modelul", command=self.start_training_thread)
        self.btn_train.grid(row=7, column=0, padx=20, pady=10)

        # 3. Verificare
        self.lbl_verify = ctk.CTkLabel(self.left_frame, text="3. Testare", text_color="gray")
        self.lbl_verify.grid(row=8, column=0, sticky="w", padx=20)

        self.btn_verify = ctk.CTkButton(self.left_frame, text="VERIFICĂ SEMNĂTURA", height=40, font=ctk.CTkFont(weight="bold"), command=self.verify_signature)
        self.btn_verify.grid(row=9, column=0, padx=20, pady=10)

        self.lbl_result = ctk.CTkLabel(self.left_frame, text="Rezultat: Așteptare...", font=ctk.CTkFont(size=16))
        self.lbl_result.grid(row=10, column=0, padx=20, pady=20)

        # PANOU DREAPTA
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        ctk.CTkLabel(self.right_frame, text="Zona de Semnătură (Mouse)", font=ctk.CTkFont(size=18)).pack(pady=10)

        self.canvas = tk.Canvas(self.right_frame, bg="white", width=500, height=300, cursor="crosshair")
        self.canvas.pack(pady=10)

        ctk.CTkButton(self.right_frame, text="Șterge Tabla", fg_color="gray", command=self.clear_canvas).pack(pady=10)

        # Logică desen
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.pil_image = Image.new("RGB", (500, 300), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self.update_counts()

    def start_draw(self, event):
        self.is_drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.is_drawing:
            x, y = event.x, event.y
            self.canvas.create_line((self.last_x, self.last_y, x, y), width=self.brush_width, fill="black", capstyle=tk.ROUND, smooth=True)
            self.pil_draw.line((self.last_x, self.last_y, x, y), fill="black", width=self.brush_width)
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.is_drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("RGB", (500, 300), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.lbl_result.configure(text="Rezultat: Așteptare...", text_color="gray")

    # --- FIX PENTRU BLOCARE NUMĂRĂTOARE ---
    def save_image(self, category):
        save_dir = os.path.join(self.dataset_path, category)
        
        # Calculăm un nume unic care SIGUR nu există
        idx = 1
        while True:
            filename = f"sig_{idx}.png"
            # Dacă fișierul nu există, îl folosim pe acesta și ieșim din buclă
            if not os.path.exists(os.path.join(save_dir, filename)):
                break
            idx += 1
            
        img_to_save = self.pil_image.resize(self.image_size).convert("L")
        img_to_save.save(os.path.join(save_dir, filename))
        
        self.update_counts()
        self.clear_canvas()
        print(f"Salvat în {category}: {filename}")

    def update_counts(self):
        n_real = len(os.listdir(os.path.join(self.dataset_path, "autentic")))
        n_fake = len(os.listdir(os.path.join(self.dataset_path, "fals")))
        
        color_real = "green" if n_real >= self.TARGET_SAMPLES else "gray"
        color_fake = "green" if n_fake >= self.TARGET_SAMPLES else "gray"
        
        self.lbl_count.configure(
            text=f"Statistici:\nAutentice: {n_real}/{self.TARGET_SAMPLES}\nFalse: {n_fake}/{self.TARGET_SAMPLES}",
            text_color=color_real if n_real >= self.TARGET_SAMPLES and n_fake >= self.TARGET_SAMPLES else "silver"
        )

    def start_training_thread(self):
        if not TF_AVAILABLE:
            self.lbl_result.configure(text="Eroare: TensorFlow lipsă!", text_color="red")
            return
        self.btn_train.configure(state="disabled", text="Se antrenează...")
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        data, labels = [], []
        
        # Încărcare date Autentic
        path_real = os.path.join(self.dataset_path, "autentic")
        for img_name in os.listdir(path_real):
            try:
                img = Image.open(os.path.join(path_real, img_name)).convert('L')
                img_arr = np.array(img) / 255.0
                data.append(img_arr.reshape(64, 64, 1))
                labels.append(1)
            except: pass

        # Încărcare date Fals
        path_fake = os.path.join(self.dataset_path, "fals")
        for img_name in os.listdir(path_fake):
            try:
                img = Image.open(os.path.join(path_fake, img_name)).convert('L')
                img_arr = np.array(img) / 255.0
                data.append(img_arr.reshape(64, 64, 1))
                labels.append(0)
            except: pass

        if len(data) < 10:
            self.lbl_result.configure(text="Prea puține date! (Min 10)", text_color="orange")
            self.finish_training_fail()
            return

        X, y = np.array(data), np.array(labels)

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
        
        print("Se antrenează...")
        model.fit(X, y, epochs=20, batch_size=8, verbose=0)
        model.save(self.model_path)
        print("Salvat!")
        self.after(0, lambda: self.finish_training_success())

    def finish_training_success(self):
        self.btn_train.configure(state="normal", text="Antrenează Modelul")
        self.lbl_result.configure(text="Model Antrenat (Succes)!", text_color="#00FF00")

    def finish_training_fail(self):
        self.btn_train.configure(state="normal", text="Antrenează Modelul")

    def verify_signature(self):
        if not TF_AVAILABLE or not os.path.exists(self.model_path):
            self.lbl_result.configure(text="Eroare: Lipsă model!", text_color="red")
            return

        try:
            model = load_model(self.model_path)
            img_to_predict = self.pil_image.resize(self.image_size).convert("L")
            img_arr = np.array(img_to_predict) / 255.0
            prediction = model.predict(img_arr.reshape(1, 64, 64, 1))
            score = prediction[0][0]
            
            if score > 0.8:
                self.lbl_result.configure(text=f"AUTENTIC ({score:.2f})", text_color="#00FF00")
            else:
                self.lbl_result.configure(text=f"FALS / SUSPECT ({score:.2f})", text_color="#FF0000")
        except:
            self.lbl_result.configure(text="Eroare la verificare", text_color="red")

if __name__ == "__main__":
    app = SignatureApp()
    app.mainloop()