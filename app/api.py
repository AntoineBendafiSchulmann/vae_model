from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import numpy as np
import onnxruntime as ort

def create_app():

    app = Flask(__name__)
    CORS(app)

    onnx_path = os.path.join("app", "models", "vae_mnist.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Modèle ONNX introuvable à {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    @app.route("/")
    def home():
        return "API Flask prête à servir le modèle ONNX pour la reconstruction VAE."

    @app.route("/image/reconstruct", methods=["POST"])
    def reconstruct_image():
        if "image" not in request.files:
            return jsonify({"error": "Aucune image envoyée."}), 400


        file = request.files["image"]
        img = np.frombuffer(file.read(), dtype=np.uint8).reshape((28, 28))
        img_normalized = img.astype(np.float32) / 255.0
        img_normalized = img_normalized.reshape(1, 1, 28, 28)

        input_tensor = {session.get_inputs()[0].name: img_normalized}
        outputs = session.run(None, input_tensor)

        reconstructed_img = outputs[0].reshape(28, 28).tolist()
        return jsonify({"reconstructed": reconstructed_img})

    return app
