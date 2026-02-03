# web_service.py
from flask import Flask, render_template_string, request, jsonify
import os
import shutil
import cv2
import base64
import numpy as np
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# IMPORTĂM modulele definite anterior
from config import init_env, FOLDERS, IMG_SIZE
from data_acquisition import pdf_to_hd_image, extract_signatures_blob_mode, get_binary_roi, get_projections, compare_structures_balanced
import neural_network  # Importam modulul complet

# Initialize
init_env()
app = Flask(__name__)

HTML_UI = """
<!DOCTYPE html>
<html lang="ro" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVAS - Sistem Verificare Semnături</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        body { background: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        .glass { background: rgba(30, 41, 59, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); }
        .btn-primary { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); transition: all 0.2s; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); }
        .btn-primary:active { transform: translateY(0); }
        .drop-zone { border: 2px dashed rgba(148, 163, 184, 0.2); transition: all 0.2s; }
        .drop-zone.dragover { border-color: #3b82f6; background: rgba(59, 130, 246, 0.1); }
        .tab-btn.active { border-bottom: 2px solid #3b82f6; color: #3b82f6; }
        .tab-btn { color: #94a3b8; }
        
        /* Scrollbar custom */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1e293b; }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #64748b; }
    </style>
</head>
<body class="min-h-screen flex flex-col items-center justify-center p-6">

    <div class="w-full max-w-5xl">
        <div class="text-center mb-8">
            <h1 class="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500 mb-2">SVAS</h1>
            <p class="text-slate-400 text-lg">Sistem Avansat de Verificare a Semnăturilor</p>
        </div>

        <div class="flex justify-center mb-8 bg-slate-900/50 p-1 rounded-lg w-max mx-auto border border-white/5">
            <button onclick="tab('enroll')" id="btn-enroll" class="tab-btn active px-8 py-3 rounded-md font-semibold transition-all flex items-center gap-2">
                <i class="fas fa-user-plus"></i> Înrolare
            </button>
            <button onclick="tab('train')" id="btn-train" class="tab-btn px-8 py-3 rounded-md font-semibold transition-all flex items-center gap-2">
                <i class="fas fa-brain"></i> Antrenare
            </button>
            <button onclick="tab('verify')" id="btn-verify" class="tab-btn px-8 py-3 rounded-md font-semibold transition-all flex items-center gap-2">
                <i class="fas fa-check-double"></i> Verificare
            </button>
        </div>

        <div class="relative min-h-[400px]">
            
            <div id="enroll" class="glass p-8 rounded-2xl transition-all duration-300">
                <div class="text-center mb-6">
                    <h2 class="text-2xl font-bold text-white mb-2">Înrolare Studenți</h2>
                    <p class="text-slate-400 text-sm">Încarcă unul sau mai multe fișiere PDF. Numele studentului va fi extras automat din numele fișierului.</p>
                </div>

                <div class="drop-zone rounded-xl p-10 text-center cursor-pointer mb-6" id="drop-enroll" onclick="document.getElementById('en_files').click()">
                    <i class="fas fa-cloud-upload-alt text-4xl text-slate-500 mb-4"></i>
                    <p class="text-lg font-medium text-slate-300">Trage fișierele aici sau click pentru a alege</p>
                    <p class="text-sm text-slate-500 mt-2">(ex: Tranca_Alexa.pdf)</p>
                    <input type="file" id="en_files" class="hidden" multiple accept=".pdf">
                </div>
                
                <div id="file-list" class="mb-4 text-sm text-slate-400 space-y-1"></div>

                <button onclick="run('enroll')" class="btn-primary w-full py-4 rounded-xl font-bold text-white shadow-lg flex justify-center items-center gap-2">
                    <i class="fas fa-bolt"></i> PROCESEAZĂ FIȘIERELE
                </button>
                <div id="en_res" class="mt-4 text-center font-medium text-slate-300 min-h-[20px]"></div>
            </div>

            <div id="train" class="hidden glass p-8 rounded-2xl text-center flex flex-col items-center justify-center min-h-[300px]">
                <div class="mb-6">
                    <div class="w-20 h-20 bg-yellow-500/10 rounded-full flex items-center justify-center mx-auto mb-4 border border-yellow-500/20">
                        <i class="fas fa-layer-group text-3xl text-yellow-500"></i>
                    </div>
                    <h2 class="text-2xl font-bold text-white">Re-Calibrare Model</h2>
                    <p class="text-slate-400 mt-2 max-w-md mx-auto">Sistemul va re-analiza geometria și stilul tuturor semnăturilor înrolate pentru a stabili noile referințe.</p>
                </div>
                <button onclick="run_training()" class="bg-yellow-600 hover:bg-yellow-500 text-white px-10 py-4 rounded-xl font-bold shadow-lg transition-all flex items-center gap-2">
                    <i class="fas fa-sync-alt"></i> START ANTRENARE
                </button>
                <div id="tr_res" class="mt-6 font-mono text-yellow-400"></div>
            </div>

            <div id="verify" class="hidden glass p-8 rounded-2xl">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                        <label class="block text-sm font-bold text-slate-400 mb-2 uppercase tracking-wider">Document de verificat (PDF)</label>
                        <div class="drop-zone rounded-xl p-8 text-center cursor-pointer h-40 flex flex-col items-center justify-center" onclick="document.getElementById('ve_file').click()">
                            <i class="fas fa-file-pdf text-3xl text-purple-500 mb-2"></i>
                            <span class="text-sm text-slate-300">Alege lista de prezență</span>
                            <input type="file" id="ve_file" class="hidden" accept=".pdf">
                        </div>
                    </div>
                    <div>
                        <label class="block text-sm font-bold text-slate-400 mb-2 uppercase tracking-wider">Lista Studenți (Nume, Prenume)</label>
                        <textarea id="ve_names" class="w-full h-40 bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-slate-200 focus:outline-none focus:border-purple-500 resize-none" placeholder="Ex: Popescu Ion, Ionescu Maria..."></textarea>
                    </div>
                </div>
                <button onclick="run('verify')" class="btn-primary bg-gradient-to-r from-purple-600 to-pink-600 w-full mt-8 py-4 rounded-xl font-bold text-white shadow-lg flex justify-center items-center gap-2">
                    <i class="fas fa-search"></i> VERIFICĂ ACUM
                </button>
                <div id="ve_res" class="mt-8 space-y-3"></div>
            </div>

        </div>
    </div>

    <script>
        // Tab Switching Logic
        function tab(t) {
            ['enroll', 'train', 'verify'].forEach(x => document.getElementById(x).classList.add('hidden'));
            document.getElementById(t).classList.remove('hidden');
            
            // Update buttons state
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active', 'border-b-2', 'border-blue-500', 'text-blue-500'));
            document.getElementById('btn-'+t).classList.add('active');
        }

        // Drag & Drop Visuals
        const dropZones = document.querySelectorAll('.drop-zone');
        dropZones.forEach(zone => {
            zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
            zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                const input = zone.querySelector('input');
                input.files = e.dataTransfer.files;
                if(input.id === 'en_files') updateFileList(input);
            });
        });

        document.getElementById('en_files').addEventListener('change', function() { updateFileList(this); });

        function updateFileList(input) {
            const list = document.getElementById('file-list');
            list.innerHTML = '';
            if (input.files.length > 0) {
                list.innerHTML = `<div class="text-blue-400 font-bold mb-1">Selectat: ${input.files.length} fișiere</div>`;
                Array.from(input.files).slice(0, 3).forEach(f => {
                    list.innerHTML += `<div><i class="fas fa-file-pdf mr-2"></i>${f.name}</div>`;
                });
                if(input.files.length > 3) list.innerHTML += `<div>... și încă ${input.files.length - 3}</div>`;
            }
        }

        async function run_training() {
            const resDiv = document.getElementById('tr_res');
            resDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Se calibrează modelul...';
            try {
                let r = await fetch('/api/train_model', {method:'POST'});
                let d = await r.json();
                resDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${d.message || d.error}`;
            } catch(e) { resDiv.innerHTML = "Eroare rețea."; }
        }

        async function run(mode) {
            let fd = new FormData();
            
            if(mode === 'enroll') {
                const files = document.getElementById('en_files').files;
                if(files.length === 0) { alert("Te rog selectează cel puțin un fișier PDF!"); return; }
                
                // Adăugăm toate fișierele în FormData
                for(let i=0; i<files.length; i++) {
                    fd.append('files[]', files[i]); // Cheia 'files[]' trimite lista
                }
                
                document.getElementById('en_res').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesare în curs...';
            } else {
                const f = document.getElementById('ve_file').files[0];
                if(!f) { alert("Selectează fișierul de verificat!"); return; }
                fd.append('file', f);
                fd.append('names', document.getElementById('ve_names').value);
                document.getElementById('ve_res').innerHTML = '<div class="text-center text-slate-400"><i class="fas fa-spinner fa-spin text-3xl"></i><br>Analiză biometrică în curs...</div>';
            }

            try {
                let r = await fetch('/api/'+mode, {method:'POST', body:fd});
                let d = await r.json();
                
                if(d.error) { 
                    alert("Eroare Server: " + d.error); 
                    if(mode=='enroll') document.getElementById('en_res').innerText = "";
                    else document.getElementById('ve_res').innerText = "";
                    return; 
                }

                if(mode === 'enroll') {
                    document.getElementById('en_res').innerHTML = `<span class="text-green-400"><i class="fas fa-check"></i> ${d.message}</span>`;
                    document.getElementById('file-list').innerHTML = ''; // Reset list
                } else {
                    let h = "";
                    d.results.forEach(res => {
                        let statusColor = res.status == 'AUTENTIC' ? "text-green-400" : (res.status == 'NESEMNAT' ? "text-yellow-500" : "text-red-500");
                        let statusIcon = res.status == 'AUTENTIC' ? "fa-check-circle" : (res.status == 'NESEMNAT' ? "fa-exclamation-circle" : "fa-times-circle");
                        
                        h += `<div class="bg-slate-800/80 p-4 rounded-xl border border-slate-700 flex items-center gap-4 hover:bg-slate-800 transition-colors">
                            <div class="w-24 h-24 bg-white rounded-lg p-2 flex items-center justify-center shrink-0">
                                <img src="${res.url}" class="max-w-full max-h-full object-contain">
                            </div>
                            <div class="flex-1 min-w-0">
                                <h3 class="font-bold text-lg text-white truncate">${res.name}</h3>
                                <div class="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-xs text-slate-400 font-mono">
                                    <div>AI Style: <span class="text-blue-400">${(res.ai_score*100).toFixed(0)}%</span></div>
                                    <div>Structură: <span class="text-purple-400">${(res.struct_score*100).toFixed(0)}%</span></div>
                                    <div>Densitate: ${(res.density_score*100).toFixed(0)}%</div>
                                    <div>Dif. Formă: ${(res.diff_ar*100).toFixed(1)}%</div>
                                </div>
                            </div>
                            <div class="text-right shrink-0">
                                <div class="text-3xl font-bold ${statusColor}">${(res.final_score*100).toFixed(0)}%</div>
                                <div class="text-xs uppercase font-bold tracking-wider text-slate-500 mt-1 flex items-center justify-end gap-1">
                                    <i class="fas ${statusIcon}"></i> ${res.status}
                                </div>
                            </div>
                        </div>`;
                    });
                    document.getElementById('ve_res').innerHTML = h;
                }
            } catch(e) { 
                alert("Eroare de comunicare: " + e); 
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def idx(): return render_template_string(HTML_UI)

@app.route('/api/train_model', methods=['POST'])
def api_train():
    return jsonify({"message": neural_network.train_custom_model()})

@app.route('/api/enroll', methods=['POST'])
def enroll():
    try:
        files = request.files.getlist('files[]')
        if not files:
            return jsonify({"error": "Niciun fișier primit!"}), 400

        total_sigs = 0
        processed_students = 0

        for f in files:
            if not f.filename: continue
            raw_name = os.path.splitext(f.filename)[0]
            name = raw_name.replace(" ", "_") 
            
            ts = datetime.now().strftime('%H%M%S')
            raw_path = os.path.join(FOLDERS["raw_sig"], f"{name}_{ts}.pdf")
            f.save(raw_path)
            
            img = pdf_to_hd_image(open(raw_path, "rb").read())
            visuals, raw_crops = extract_signatures_blob_mode(img, is_grid=True, expected_count=20)
            
            p_dir = os.path.join(FOLDERS["processed_sig"], name)
            t_dir = os.path.join(FOLDERS["train"], name)
            
            if os.path.exists(p_dir): shutil.rmtree(p_dir)
            if os.path.exists(t_dir): shutil.rmtree(t_dir)
            os.makedirs(p_dir); os.makedirs(t_dir)

            count = 0
            for i, (vis, raw) in enumerate(zip(visuals, raw_crops)):
                if raw is not None:
                    roi, _ = get_binary_roi(raw)
                    if roi is not None and cv2.countNonZero(roi) > 50:
                        cv2.imwrite(os.path.join(p_dir, f"{i+1}.png"), vis)
                        cv2.imwrite(os.path.join(t_dir, f"{i+1}.png"), vis)
                        count += 1
            
            if count > 0:
                total_sigs += count
                processed_students += 1

        return jsonify({"message": f"Succes! {processed_students} studenți înrolați ({total_sigs} semnături total)."})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify():
    try:
        f = request.files['file']
        names = [x.strip() for x in request.form['names'].split(',') if x.strip()]
        ts = datetime.now().strftime('%H%M%S')
        path = os.path.join(FOLDERS["raw_list"], f"list_{ts}.pdf")
        with open(path, "wb") as wb: wb.write(f.read())
        img = pdf_to_hd_image(open(path, "rb").read())
        
        visuals, raw_crops = extract_signatures_blob_mode(img, is_grid=False, expected_count=len(names))
        res = []
        p_dir = os.path.join(FOLDERS["processed_list"], ts)
        os.makedirs(p_dir, exist_ok=True)
        
        for i, (vis, raw) in enumerate(zip(visuals, raw_crops)):
            if i >= len(names): break
            name = names[i]
            cv2.imwrite(os.path.join(p_dir, f"{name}.png"), vis)
            _, b = cv2.imencode('.png', vis)
            url = f"data:image/png;base64,{base64.b64encode(b).decode()}"
            
            ai_score, ar_score, density_score, proj_score, final_score = 0, 0, 0, 0, 0
            struct_score = 0
            diff_ar = 0.0
            st = "UNKNOWN"
            
            if name in neural_network.SIGNATURE_DATA:
                roi, ar = get_binary_roi(raw)
                if roi is None or cv2.countNonZero(roi) < 50: 
                    st = "NESEMNAT"
                else:
                    rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
                    batch = preprocess_input(rgb.astype(np.float32))
                    vec = neural_network.feature_extractor.predict(np.expand_dims(batch, 0), verbose=0).flatten()
                    sims = [neural_network.get_similarity_score(vec, r) for r in neural_network.SIGNATURE_DATA[name]['vectors']]
                    ai_score = float(np.mean(sorted(sims, reverse=True)[:3]))

                    proj = get_projections(roi)
                    best_match = (0, 0, 0, 1.0)
                    
                    for (ref_roi, ref_ar, ref_proj) in neural_network.SIGNATURE_DATA[name]['geo_refs']:
                        s_ar, s_den, s_proj, d_ar = compare_structures_balanced(roi, ar, proj, ref_roi, ref_ar, ref_proj, ai_score)
                        curr_total = s_ar + s_den + s_proj
                        best_total = best_match[0] + best_match[1] + best_match[2]
                        if curr_total > best_total:
                            best_match = (s_ar, s_den, s_proj, d_ar)
                    
                    ar_score, density_score, proj_score, diff_ar = best_match
                    struct_score = (ar_score * 0.2) + (density_score * 0.4) + (proj_score * 0.4)
                    
                    # Logică de calcul a scorului
                    if ai_score > 0.90 and struct_score > 0.50:
                        final_score = (ai_score * 0.6) + (struct_score * 0.4)
                    elif struct_score > 0.85:
                        final_score = (ai_score * 0.3) + (struct_score * 0.7)
                    else:
                        final_score = (ai_score * 0.4) + (struct_score * 0.6)
                    
                    # [MODIFICARE] Penalizare dacă structura e slabă (<78%)
                    if struct_score < 0.78:
                        final_score -= 0.15 # Scădem 15% din scor
                    
                    # [MODIFICARE NOUĂ] Penalizare pentru diferența de formă (Aspect Ratio)
                    if diff_ar > 0.20:
                        final_score -= 0.15 # Scădem încă 15% dacă forma diferă mult
                        
                    if final_score < 0: final_score = 0

                    if 0.88 <= final_score < 0.90 and diff_ar < 0.1:
                        final_score = 0.90

                    st = "AUTENTIC" if final_score >= 0.90 else "SUSPECT"

            res.append({
                "name": name, "final_score": final_score, "ai_score": ai_score,
                "struct_score": struct_score, "density_score": density_score,
                "diff_ar": diff_ar, "status": st, "url": url
            })
            
        return jsonify({"results": res})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    neural_network.load_ai()
    app.run(debug=False, port=5000)