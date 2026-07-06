import os
import io
from flask import Flask, render_template_string, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# 1. கொலாபில் உங்களின் அக்யூரேட் ரிசல்ட் கொடுத்த 380x380 பெஸ்ட் மாடலை லோடு செய்கிறோம்
try:
    model_path = r"C:\Users\Placement\Music\original deepfake\efficientnet_deepfake_best.keras"
    
    # மாடலை லோடு செய்கிறோம்
    model = tf.keras.models.load_model(model_path)
    print("\n" + "="*60)
    print(">>> 🚀 SUCCESS: 380x380 BEST MODEL SUCCESSFULLY LOADED! <<<")
    print("="*60 + "\n")
except Exception as e:
    print("\n" + "="*60)
    print(f">>> ❌ ERROR LOADING BEST MODEL: {e}")
    print("தயவுசெய்து 'efficientnet_deepfake_best.keras' ஃபைல் அதே ஃபோல்டரில் உள்ளதா என பாருங்கள்!")
    print("="*60 + "\n")

# 2. HTML வெப்சைட் டிசைன்
HTML_CODE = """
<!DOCTYPE html>
<html lang="ta">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake கண்டறியும் தளம்</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f0f4f8; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .container { background-color: white; padding: 30px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); width: 420px; text-align: center; }
        h2 { color: #1e293b; margin-bottom: 10px; font-size: 24px; }
        .subtitle { color: #64748b; font-size: 14px; margin-bottom: 25px; }
        .upload-box { border: 2px dashed #cbd5e1; padding: 30px 20px; border-radius: 12px; cursor: pointer; background-color: #f8fafc; transition: all 0.3s ease; }
        .upload-box:hover { border-color: #3b82f6; background-color: #eff6ff; }
        .upload-box p { margin: 0; color: #64748b; font-size: 15px; }
        #imagePreview { max-width: 100%; max-height: 180px; margin-top: 15px; border-radius: 8px; display: none; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
        button { margin-top: 20px; background-color: #3b82f6; color: white; border: none; padding: 14px; font-size: 16px; border-radius: 8px; cursor: pointer; width: 100%; font-weight: bold; transition: background 0.2s; }
        button:hover { background-color: #2563eb; }
        #resultContainer { margin-top: 25px; padding: 15px; border-radius: 8px; display: none; font-size: 16px; font-weight: bold; }
        .real { background-color: #dcfce7; color: #16a34a; border: 1px solid #bbf7d0; }
        .fake { background-color: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }
        #loading { display: none; color: #3b82f6; margin-top: 15px; font-weight: bold; }
    </style>
</head>
<body>
<div class="container">
    <h2>🛡️ Deepfake Detection Portal</h2>
    <p class="subtitle">புகைப்படத்தை அப்لوடு செய்து அது உண்மையா அல்லது போலியா என கண்டறியவும்.</p>
    <div class="upload-box" onclick="document.getElementById('fileInput').click()">
        <p>இங்கே கிளிக் செய்து புகைப்படத்தை தேர்ந்தெடுக்கவும்</p>
        <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="previewImage(event)">
        <center><img id="imagePreview" alt="Preview"></center>
    </div>
    <button onclick="uploadAndPredict()">பரிசோதிக்கவும் (Analyze)</button>
    <div id="loading">பரிசோதிக்கப்படுகிறது... சற்று காத்திருக்கவும்...</div>
    <div id="resultContainer"></div>
</div>
<script>
    function previewImage(event) {
        const reader = new FileReader();
        reader.onload = function(){
            const output = document.getElementById('imagePreview');
            output.src = reader.result;
            output.style.display = 'block';
        };
        reader.readAsDataURL(event.target.files[0]);
        document.getElementById('resultContainer').style.display = 'none';
    }
    async function uploadAndPredict() {
        const fileInput = document.getElementById('fileInput');
        const resultContainer = document.getElementById('resultContainer');
        const loading = document.getElementById('loading');
        if (fileInput.files.length === 0) { alert("தயவுசெய்து ஒரு புகைப்படத்தை முதலில் தேர்ந்தெடுக்கவும்!"); return; }
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        loading.style.display = 'block';
        resultContainer.style.display = 'none';
        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            loading.style.display = 'none';
            if (data.error) { alert("பிழை: " + data.error); return; }
            resultContainer.innerHTML = `${data.result} <br> <span style="font-size:13px; font-weight:normal; color:#475569;">உறுதித்தன்மை (Confidence): ${data.confidence}</span>`;
            resultContainer.style.display = 'block';
            if (data.status === "real") { resultContainer.className = "real"; } else { resultContainer.className = "fake"; }
        } catch (error) { loading.style.display = 'none'; alert("பரிசோதிப்பதில் பிழை ஏற்பட்டது."); console.error(error); }
    }
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_CODE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Entha file-um upload seiyappadavillai'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File தேர்ந்தெடுக்கப்படவில்லை'}), 400
    try:
        # 1. இமேஜ் பைட்டுகளை ரீட் செய்தல்
        img_bytes = file.read()
        
        # 2. கொலாப் ஸ்டைல் டென்சர்ஃப்ளோ ப்ரீபிராசஸிங்
        img_tensor = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        
        # 3. உங்கள் அசல் பெஸ்ட் மாடலுக்குத் தேவையான 380x380 இன்புட் சைஸ்!
        img_tensor = tf.image.resize(img_tensor, [380, 380])
        
        img_array = tf.cast(img_tensor, tf.float32)
        img_array = tf.expand_dims(img_array, axis=0) # (1, 380, 380, 3)

        # 4. மாடல் பிரிடிக்சன்
        predictions = model.predict(img_array)
        score = float(predictions[0][0])
        
        print(f"\n>>>> 🎯 BEST MODEL RAW SCORE: {score} <<<<\n")

        # 5. கொலாப் அக்யூரசி லாஜிக்:
        # உங்கள் மாடலின் படி score 0.5-க்கு மேல் இருந்தால் REAL (உண்மையான படம்)
        # ஒருவேளை அவுட்புட் உல்டாவாக வந்தால் மட்டும் 'score >= 0.5' என்பதை 'score < 0.5' என மாற்றிக் கொள்ளலாம்.
        if score >= 0.5:
            result = "உண்மையான படம் (REAL IMAGE)"
            confidence_score = score
            status = "real"
        else:
            result = "போலி படம் (FAKE DEEPFAKE)"
            confidence_score = 1 - score
            status = "fake"

        confidence_percent = f"{confidence_score * 100:.2f}%"

        return jsonify({'result': result, 'confidence': confidence_percent, 'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=9000)
