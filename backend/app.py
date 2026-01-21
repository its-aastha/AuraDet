from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "AuraDet Backend is Running"

@app.route("/predict", methods=["POST"])
def predict_emotion():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    try:
        result = DeepFace.analyze(
            image,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]
        confidence = result[0]["emotion"][emotion]

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
