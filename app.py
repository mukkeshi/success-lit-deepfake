import os
import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template_string

# Reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

model = None
model_error = None

# Keras 3 Serialization Patch
try:
    from keras.src.saving import serialization_lib
    original_deserialize = serialization_lib.deserialize_keras_object
    
    def patched_deserialize(config, *args, **kwargs):
        if isinstance(config, dict):
            if config.get('class_name') == 'Dense' and 'config' in config:
                config['config'].pop('quantization_config', None)
            if 'config' in config and isinstance(config['config'], dict):
                inner_config = config['config']
                if 'layers' in inner_config:
                    for layer in inner_config['layers']:
                        if layer.get('class_name') == 'Dense' and 'config' in layer:
                            layer['config'].pop('quantization_config', None)
        return original_deserialize(config, *args, **kwargs)
        
    serialization_lib.deserialize_keras_object = patched_deserialize
except Exception as patch_err:
    print(f"Patch apply failed: {patch_err}")

# Load Deepfake Model
try:
    model_path = 'efficientnet_deepfake_best.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    else:
        model_error = "Model file ('efficientnet_deepfake_best.keras') not found!"
except Exception as e:
    model_error = str(e)

# --- PURE ENGLISH HIGH-TECH UI ---
HTML_CODE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DEEP EYE // AI DETECTOR</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #06070d;
            --card-bg: #0d0f1d;
            --neon-blue: #00f2fe;
            --neon-purple: #9b51e0;
            --neon-green: #39ff14;
            --neon-red: #ff3131;
            --text-main: #ffffff;
            --text-sub: #787f9d;
        }

        body {
            font-family: 'Rajdhani', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background-image: radial-gradient(circle at 50% 50%, #11142a 0%, #06070d 90%);
        }

        .dashboard {
            width: 90%;
            max-width: 700px;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 0 40px rgba(0, 242, 254, 0.1);
            border: 1px solid rgba(0, 242, 254, 0.15);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
        }

        .dashboard::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 3px;
            background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple));
        }

        h1 {
            font-family: 'Orbitron', sans-serif;
            font-weight: 900;
            font-size: 2.5rem;
            margin: 0;
            letter-spacing: 2px;
            background: linear-gradient(45deg, var(--neon-blue), #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
        }

        .tagline {
            color: var(--text-sub);
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-top: 5px;
            margin-bottom: 40px;
        }

        .upload-box {
            border: 2px dashed rgba(0, 242, 254, 0.25);
            padding: 40px 20px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.01);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            margin-bottom: 30px;
            text-align: center;
        }

        .upload-box:hover {
            border-color: var(--neon-blue);
            background: rgba(0, 242, 254, 0.03);
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.1);
        }

        .upload-icon {
            font-size: 3rem;
            color: var(--neon-blue);
            margin-bottom: 15px;
        }

        .upload-text {
            font-size: 1.3rem;
            font-weight: 600;
            color: #b0b7d6;
        }

        .upload-subtext {
            font-size: 0.95rem;
            color: var(--text-sub);
            margin-top: 5px;
        }

        input[type="file"] {
            display: none;
        }

        /* Preview Panel */
        #preview-container {
            display: none;
            margin: 20px auto;
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
            background: #000;
        }

        #image-preview, #video-preview {
            max-width: 100%;
            max-height: 300px;
            display: block;
            margin: 0 auto;
        }

        .btn-core {
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            background: linear-gradient(45deg, var(--neon-blue), var(--neon-purple));
            color: #000;
            border: none;
            padding: 16px;
            width: 100%;
            border-radius: 8px;
            cursor: pointer;
            letter-spacing: 2px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.3);
        }

        .btn-core:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 30px rgba(0, 242, 254, 0.5);
            filter: brightness(1.1);
        }

        /* Results UI */
        .result-panel {
            margin-top: 40px;
            padding: 25px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            text-align: left;
        }

        .result-header {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-real { color: var(--neon-green); text-shadow: 0 0 15px rgba(57, 255, 20, 0.3); }
        .status-fake { color: var(--neon-red); text-shadow: 0 0 15px rgba(255, 49, 49, 0.3); }

        .meta-text {
            font-size: 1.1rem;
            color: var(--text-sub);
            margin-bottom: 10px;
        }

        .bar-bg {
            background: rgba(255, 255, 255, 0.08);
            height: 12px;
            border-radius: 20px;
            overflow: hidden;
            margin-top: 8px;
        }

        .bar-fill {
            height: 100%;
            border-radius: 20px;
            width: 0;
            transition: width 1s cubic-bezier(0.1, 0.8, 0.2, 1);
        }

        .fill-real { background: linear-gradient(90deg, #2ecc71, var(--neon-green)); box-shadow: 0 0 15px var(--neon-green); }
        .fill-fake { background: linear-gradient(90deg, #e74c3c, var(--neon-red)); box-shadow: 0 0 15px var(--neon-red); }
    </style>
</head>
<body>

<div class="dashboard">
    <h1>DEEP EYE AI</h1>
    <div class="tagline">Next-Gen Media Authentication Terminal</div>

    <form action="/" method="POST" enctype="multipart/form-data">
        <div class="upload-box" onclick="document.getElementById('file-input').click()">
            <div class="upload-icon">📂</div>
            <div class="upload-text" id="box-main-text">Drop Media or Click to Browse</div>
            <div class="upload-subtext">Supports Images & Videos (MP4, AVI)</div>
            <input id="file-input" type="file" name="file" accept="image/*,video/*" required onchange="handleFileSelect(this)">
        </div>

        <div id="preview-container">
            <img id="image-preview" src="#" alt="Image Preview" style="display:none;">
            <video id="video-preview" controls style="display:none;"></video>
        </div>

        <button type="submit" class="btn-core">ANALYZE SOURCE</button>
    </form>

    {% if result %}
    <div class="result-panel">
        <div class="meta-text">Type: <strong>{{ media_type }}</strong></div>
        
        {% if "REAL" in result %}
            <div class="result-header status-real">⚡ ANALYSIS: {{ result }}</div>
            <div class="meta-text">AI Confidence Score: <strong>{{ confidence }}%</strong></div>
            <div class="bar-bg">
                <div class="bar-fill fill-real" style="width: {{ confidence }}%"></div>
            </div>
        {% else %}
            <div class="result-header status-fake">🚨 ANALYSIS: {{ result }}</div>
            <div class="meta-text">AI Confidence Score: <strong>{{ confidence }}%</strong></div>
            <div class="bar-bg">
                <div class="bar-fill fill-fake" style="width: {{ confidence }}%"></div>
            </div>
        {% endif %}
    </div>
    {% endif %}
</div>

<script>
function handleFileSelect(input) {
    const mainText = document.getElementById('box-main-text');
    const container = document.getElementById('preview-container');
    const imgPrev = document.getElementById('image-preview');
    const vidPrev = document.getElementById('video-preview');

    if (input.files && input.files[0]) {
        const file = input.files[0];
        mainText.textContent = "Selected: " + file.name;
        container.style.display = "block";

        const reader = new FileReader();

        if (file.type.startsWith('image/')) {
            vidPrev.style.display = "none";
            vidPrev.src = "";
            reader.onload = function(e) {
                imgPrev.src = e.target.result;
                imgPrev.style.display = "block";
            }
            reader.readAsDataURL(file);
        } else if (file.type.startsWith('video/')) {
            imgPrev.style.display = "none";
            imgPrev.src = "#";
            vidPrev.src = URL.createObjectURL(file);
            vidPrev.style.display = "block";
            vidPrev.load();
        }
    }
}
</script>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if model_error:
            return render_template_string(HTML_CODE, result=f"Model Initialization Error: {model_error}", confidence="0", media_type="Unknown")
        if model is None:
            return render_template_string(HTML_CODE, result="AI Core Offline", confidence="0", media_type="Unknown")
            
        if 'file' not in request.files:
            return render_template_string(HTML_CODE, result="Payload Missing", confidence="0", media_type="Unknown")
            
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_CODE, result="Empty Selection", confidence="0", media_type="Unknown")

        filename = file.filename.lower()
        
        # --- VIDEO DETECTING LOGIC ---
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                # Save temp video file
                temp_path = "temp_uploaded_video.mp4"
                file.save(temp_path)
                
                cap = cv2.VideoCapture(temp_path)
                scores = []
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Extract 1 frame every 15 frames to speed up processing
                    if frame_count % 15 == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_tensor = tf.image.resize(rgb_frame, [380, 380])
                        img_array = tf.cast(img_tensor, tf.float32)
                        img_array = tf.expand_dims(img_array, axis=0)
                        
                        predictions = model.predict(img_array, verbose=0)
                        scores.append(float(predictions[0][0]))
                        
                    frame_count += 1
                cap.release()
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                if not scores:
                    return render_template_string(HTML_CODE, result="Failed to read video frames", confidence="0", media_type="Video")
                
                # Average score logic
                avg_score = sum(scores) / len(scores)
                if avg_score >= 0.5:
                    result = "REAL VIDEO"
                    confidence_str = f"{avg_score * 100:.2f}"
                else:
                    result = "DEEPFAKE VIDEO"
                    confidence_str = f"{(1 - avg_score) * 100:.2f}"
                    
                return render_template_string(HTML_CODE, result=result, confidence=confidence_str, media_type="Video")
                
            except Exception as e:
                return render_template_string(HTML_CODE, result=f"Video Pipeline Error: {str(e)}", confidence="0", media_type="Video")

        # --- IMAGE DETECTING LOGIC ---
        else:
            try:
                img_bytes = file.read()
                img_tensor = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
                img_tensor = tf.image.resize(img_tensor, [380, 380])
                img_array = tf.cast(img_tensor, tf.float32)
                img_array = tf.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array)
                score = float(predictions[0][0])
                
                if score >= 0.5:
                    result = "REAL IMAGE"
                    confidence_str = f"{score * 100:.2f}"
                else:
                    result = "DEEPFAKE IMAGE"
                    confidence_str = f"{(1 - score) * 100:.2f}"
                    
                return render_template_string(HTML_CODE, result=result, confidence=confidence_str, media_type="Image")
                
            except Exception as e:
                return render_template_string(HTML_CODE, result=f"Image Scan Error: {str(e)}", confidence="0", media_type="Image")
                
    return render_template_string(HTML_CODE, result=None, confidence="0", media_type=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
