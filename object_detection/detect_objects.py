import cv2
import numpy as np
import tensorflow as tf
import time

# Load the TensorFlow SavedModel
MODEL_DIR = "saved_model"
model = tf.saved_model.load(MODEL_DIR)
detect_fn = model.signatures["serving_default"]

# COCO class labels
COCO_LABELS = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 5: "Airplane",
    6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic Light",
    11: "Fire Hydrant", 13: "Stop Sign", 14: "Parking Meter", 15: "Bench",
    16: "Bird", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep",
    21: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra", 25: "Giraffe"
}

# Object height estimates in meters (for distance estimation)
REAL_HEIGHTS = {
    "Person": 1.7, "Car": 1.5, "Bicycle": 1.0, "Motorcycle": 1.2,
    "Bus": 3.0, "Truck": 3.5, "Traffic Light": 2.5
}

# Focal length can be calibrated dynamically
DEFAULT_FOCAL_LENGTH = 600  # Default value, should be adjusted per camera
FOCAL_LENGTH = DEFAULT_FOCAL_LENGTH  

def calibrate_focal_length(known_object_height, pixel_height, real_height):
    """Calculate focal length dynamically based on a known object."""
    return (pixel_height * known_object_height) / real_height

def estimate_distance(bbox, image_height, label):
    """Estimate distance of an object based on its bounding box height."""
    ymin, ymax = bbox  # Normalized coordinates (0 to 1)

    # Convert to absolute pixel coordinates
    ymin, ymax = int(ymin * image_height), int(ymax * image_height)
    object_height = ymax - ymin  

    # Avoid division errors
    if object_height <= 0:
        return 10.0  # Assign a default maximum distance

    # Use predefined real-world height or default to 1m
    real_height = REAL_HEIGHTS.get(label, 1.0)  

    # Compute estimated distance
    estimated_distance = (real_height * FOCAL_LENGTH) / object_height
    return round(estimated_distance, 2)  # Return in meters

def detect_objects_and_distance(frame):
    """Detect objects in the frame and estimate their distances."""
    global FOCAL_LENGTH  # Allow dynamic focal length adjustment
    object_counts = {}  # Reset per frame

    h, w, _ = frame.shape

    # Convert frame to tensor and run inference
    input_tensor = tf.convert_to_tensor(frame)[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # Extract detection results
    boxes = detections['detection_boxes'].numpy()[0]  
    classes = detections['detection_classes'].numpy()[0].astype(int)  
    scores = detections['detection_scores'].numpy()[0]  

    detected_objects = []

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = classes[i]
            label = COCO_LABELS.get(class_id, f"Unknown ({class_id})")
            bbox = boxes[i][:4]  # Normalized (ymin, xmin, ymax, xmax)

            # Assign unique labels like "Person 1", "Car 2"
            object_counts[label] = object_counts.get(label, 0) + 1
            unique_label = f"{label} {object_counts[label]}"

            # Estimate distance using bbox height
            distance = estimate_distance((bbox[0], bbox[2]), h, label)

            # Convert bbox to absolute pixel coordinates
            ymin, xmin, ymax, xmax = (bbox * [h, w, h, w]).astype(int)

            detected_objects.append((unique_label, distance, (xmin, ymin, xmax, ymax)))

    # Add a delay (adjust as needed)
    time.sleep(0.1)  # 100ms delay

    return detected_objects
