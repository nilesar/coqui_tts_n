from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import os
import tempfile
import numpy as np
import soundfile as sf
import requests
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

# 📥 Function to download from Google Drive
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

# 🚀 Load TTS model
def load_tts_model(model_path, config_path):
    try:
        print("Loading config...")
        config = load_config(config_path)

        print("Initializing model...")
        model = Vits.init_from_config(config)

        print("Loading weights...")
        model.load_checkpoint(config, checkpoint_path=model_path)
        model.eval()

        print("Model ready ✅")
        return model
    except Exception as e:
        print("Model load failed:", str(e))
        raise

# 🔁 Initialize model
model = load_tts_model(model_path, config_path)

# 🎙️ Synthesize endpoint
@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        print(f"Synthesizing: {text}")
        processed_text = model.tokenizer.text_to_ids(text)
        processed_text = torch.tensor(processed_text).unsqueeze(0)

        model_outputs = model.inference(processed_text)
        waveform = model_outputs.get('model_outputs')

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze(0).cpu().numpy()

        sample_rate = model_outputs.get('sample_rate', 22050)

        output_path = os.path.join(output_dir, "output6.wav")
        sf.write(output_path, waveform, samplerate=sample_rate, format='WAV')

        return jsonify({"audio_url": "/output6.wav"})

    except Exception as e:
        print("Synthesis failed:", str(e))
        return jsonify({"error": str(e)}), 500

# 🔊 Serve audio
@app.route('/output6.wav')
def serve_audio():
    return send_file(os.path.join(output_dir, "output6.wav"), mimetype="audio/wav")

# 🔥 No app.run() block needed — Gunicorn will run the app
