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
    print("\n>>> 🚀 SUCCESS: MODEL LOADED! <<<\n")
except Exception as e:
    print(f"\n>>> ❌ ERROR LOADING MODEL: {e}\n")

# 2. எளிமையான HTML டிசைன் (வெள்ளை ஸ்கிரீன் எர்ரர் வராமல் தடுக்க)
HTML_CODE = """
<!DOCTYPE html>
<html lang="ta">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detection</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background: #f4f4f9; }
        .container { background: white; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .result { margin-top: 20px; font-weight: bold; font-size: 1.2em; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Deepfake Image Detection System</h2>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br><br>
            <button type="submit">Check Image</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_CODE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        img_tensor = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img_tensor = tf.image.resize(img_tensor, [380, 380])
        img_array = tf.cast(img_tensor, tf.float32)
        img_array = tf.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        score = float(predictions[0][0])
        
        if score >= 0.5:
            result = "உண்மையான படம் (REAL IMAGE)"
        else:
            result = "போலியான படம் (DEEPFAKE IMAGE)"
            
        return f"<h3>Result: {result}</h3><p>Confidence Score: {score:.4f}</p><br><a href='/'>Back to Home</a>"
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
