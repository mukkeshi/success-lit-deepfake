import os
import io
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# 1. மாடல் பாதையை சர்வர் மற்றும் லோக்கலுக்கு ஏத்தபடி மாற்றியுள்ளோம்
try:
    model_path = "efficientnet_deepfake_best.keras"
    model = tf.keras.models.load_model(model_path)
    print("\n>>> 🚀 SUCCESS: MODEL LOADED SUCCESSFULLY! <<<\n")
except Exception as e:
    print(f"\n>>> ❌ ERROR LOADING MODEL: {e}\n")

# 2. உங்கள் அசல் முழுமையான HTML டிசைன் (ரிசல்ட் பாக்ஸ் உடன்)
HTML_CODE = """
<!DOCTYPE html>
<html lang="ta">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake கண்டறியும் கருவி</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #333;
        }
        .container {
            background: #fff;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            text-align: center;
            max-width: 450px;
            width: 100%;
        }
        h2 {
            color: #4a5568;
            margin-bottom: 10px;
            font-weight: 600;
        }
        p.subtitle {
            color: #718096;
            font-size: 14px;
            margin-bottom: 30px;
        }
        .upload-box {
            border: 2px dashed #cbd5e0;
            padding: 30px 20px;
            border-radius: 15px;
            background: #f7fafc;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-box:hover {
            border-color: #667eea;
            background: #edf2f7;
        }
        input[type="file"] {
            display: block;
            margin: 0 auto 20px auto;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(118, 75, 162, 0.3);
            transition: transform 0.2s ease;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 18px;
            animation: fadeIn 0.5s ease;
        }
        .real {
            background-color: #c6f6d5;
            color: #22543d;
            border: 1px solid #9ae6b4;
        }
        .fake {
            background-color: #fed7d7;
            color: #742a2a;
            border: 1px solid #feb2b2;
        }
        .score {
            font-size: 14px;
            font-weight: normal;
            margin-top: 5px;
            color: #4a5568;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Deepfake Image Detection</h2>
        <p class="subtitle">AI மூலம் புகைப்படங்களின் உண்மைத்தன்மையை கண்டறியவும்</p>
        
        <form method="POST" action="/" enctype="multipart/form-data">
            <div class="upload-box">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">பரிசோதிக்கவும் (Check)</button>
            </div>
        </form>

        {% if result %}
            {% if "REAL" in result %}
                <div class="result-box real">
                    🎯 முடிவுகள்: {{ result }}
                    <div class="score">உறுதித்தன்மை (Confidence): {{ confidence }}%</div>
                </div>
            {% else %}
                <div class="result-box fake">
                    ⚠️ முடிவுகள்: {{ result }}
                    <div class="score">உறுதித்தன்மை (Confidence): {{ confidence }}%</div>
                </div>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence_str = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_CODE, result="Error: No file uploaded", confidence="0")
            
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_CODE, result="Error: No file selected", confidence="0")

        try:
            img_bytes = file.read()
            # டென்சர்ஃப்ளோ ப்ரீபிராசஸிங் (380x380 இன்புட் சைஸ்)
            img_tensor = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            img_tensor = tf.image.resize(img_tensor, [380, 380])
            img_array = tf.cast(img_tensor, tf.float32)
            img_array = tf.expand_dims(img_array, axis=0)

            # பிரிடிக்சன் செய்தல்
            predictions = model.predict(img_array)
            score = float(predictions[0][0])
            
            # கொலாப் அக்யூரசி லாஜிக் படி பிரித்தல்
            if score >= 0.5:
                result = "உண்மையான படம் (REAL IMAGE)"
                confidence_str = f"{score * 100:.2f}"
            else:
                result = "போலியான படம் (DEEPFAKE IMAGE)"
                confidence_str = f"{(1 - score) * 100:.2f}"
                
        except Exception as e:
            result = f"Prediction failed: {str(e)}"
            confidence_str = "0"

    return render_template_string(HTML_CODE, result=result, confidence=confidence_str)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
