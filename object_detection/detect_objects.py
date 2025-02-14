import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, List

# Load the MobileNet SSD Model
MODEL_DIR = "saved_model"
model = tf.saved_model.load(MODEL_DIR)
detect_fn = model.signatures["serving_default"]

# COCO Class Labels
COCO_LABELS = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 5: "Airplane",
    6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic Light",
    11: "Fire Hydrant", 13: "Stop Sign", 14: "Parking Meter", 15: "Bench",
    16: "Bird", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep",
    21: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra", 25: "Giraffe"
}

def estimate_distance(bbox: Tuple[float, float, float, float], image_height: int) -> float:
    """ Estimate actual distance using a scaling factor. """
    ymin, xmin, ymax, xmax = bbox
    object_height = ymax - ymin
    estimated_distance = max(0.1, min(10, 2.0 / object_height))  # Normalize initial estimation

    # Apply correction factor
    corrected_distance = estimated_distance * (0.30 / 2.60)  # Scale down
    return round(corrected_distance, 2)

def detect_objects_and_distance(camera_input: cv2.VideoCapture) -> List[Tuple[str, float]]:
    """ Detect objects and estimate their distances. """
    ret, frame = camera_input.read()
    if not ret:
        return []

    h, w, _ = frame.shape

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the model
    detections = detect_fn(input_tensor)

    # Extract detection data
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(int)
    scores = detections['detection_scores'].numpy()[0]

    results = []
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = classes[i]
            label = COCO_LABELS.get(class_id, f"Unknown ({class_id})")
            distance = estimate_distance(boxes[i], h)
            results.append((label, distance))

    return results
