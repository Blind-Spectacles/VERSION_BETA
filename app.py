from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from object_detection.detect_objects import detect_objects_and_distance
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for frontend/mobile integration

# Load the object detection model
MODEL_DIR = "saved_model"
model = tf.saved_model.load(MODEL_DIR)
detect_fn = model.signatures["serving_default"]

@app.route("/")
def home():
    return jsonify({"message": "Object Detection API is running!"})

@app.route("/detect", methods=["POST"])
def detect_objects():
    """ Endpoint to process an image and return detected objects with distances """
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["file"]
    image = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = detect_objects_and_distance(frame)
    
    return jsonify({"detections": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
