from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import os
import tempfile
import numpy as np
import soundfile as sf
import requests
import uuid
import gc

from TTS.config import load_config
from TTS.tts.models.vits import Vits

app = Flask(__name__)
CORS(app)

# 🔗 Google Drive Direct Download Links
MODEL_URL = "https://drive.google.com/uc?export=download&id=1D5uHC9lK4c0dK5-vPGHY-mTSVAIq97QX"
CONFIG_URL = "https://drive.google.com/uc?export=download&id=1U3MKG8n0XlxIx-w6HX65hwES3i5IG56n"

# 🔧 Local filenames to save the downloads
model_path = "model_file.pth"
config_path = "config.json"

# 📥 Download from Google Drive if not already present
def download_if_not_exists(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {save_path} from Google Drive...")
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {save_path}")

# ⬇️ Download model + config
download_if_not_exists(MODEL_URL, model_path)
download_if_not_exists(CONFIG_URL, config_path)

# 🔊 Output directory
output_dir = tempfile.mkdtemp()

# 🔁 Lazy model loader
model = None
def get_model():
    global model
    if model is None:
        try:
            print("Loading config...")
            config = load_config(config_path)

            print("Initializing model...")
            model_instance = Vits.init_from_config(config)

            print("Loading weights...")
            model_instance.load_checkpoint(config, checkpoint_path=model_path)
            model_instance.eval()
            model_instance.to("cpu")  # Force CPU

            print("Model ready ✅")
            model = model_instance
        except Exception as e:
            print("Model load failed:", str(e))
            raise
    return model

# 🎙️ Synthesize endpoint
@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        print(f"Synthesizing: {text}")
        model_instance = get_model()

        processed_text = model_instance.tokenizer.text_to_ids(text)
        processed_text = torch.tensor(processed_text).unsqueeze(0)

        model_outputs = model_instance.inference(processed_text)
        waveform = model_outputs.get('model_outputs')

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze(0).cpu().numpy()

        sample_rate = model_outputs.get('sample_rate', 22050)

        filename = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, waveform, samplerate=sample_rate, format='WAV')

        # Clean up unused memory
        gc.collect()

        return jsonify({"audio_url": f"/audio/{filename}"})

    except Exception as e:
        print("Synthesis failed:", str(e))
        return jsonify({"error": str(e)}), 500

# 🔊 Serve audio dynamically
@app.route('/audio/<filename>')
def serve_audio(filename):
    file_path = os.path.join(output_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/wav")
    return jsonify({"error": "File not found"}), 404
