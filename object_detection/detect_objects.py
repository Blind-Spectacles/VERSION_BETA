import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow SavedModel
MODEL_DIR = "VERSION_BETA-final-model/saved_model"
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

# Dictionary to track object counts
object_counts = {}

def estimate_distance(bbox, image_height):
    """Estimate distance based on object size in the frame."""
    ymin, xmin, ymax, xmax = bbox  # These are normalized (0 to 1)

    # Convert to absolute pixel coordinates
    ymin, ymax = int(ymin * image_height), int(ymax * image_height)

    # Compute object height in pixels
    object_height = ymax - ymin  

    # Ensure the height is a valid positive number
    if object_height <= 0:
        return 10.0  # Assign a default max distance if invalid

    # Camera calibration parameters (tune these for your setup)
    focal_length = 600  # Adjust for your camera
    real_height = 1.7  # Average height of a person in meters

    estimated_distance = (real_height * focal_length) / object_height

    return round(estimated_distance, 2)  # Return in meters
  # Round to 2 decimal places


def detect_objects_and_distance(frame):
    """Detect objects using TensorFlow and estimate distances."""
    global object_counts
    object_counts = {}  # Reset the dictionary each frame


    h, w, _ = frame.shape

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run model inference
    detections = detect_fn(input_tensor)

    # Extract detection data
    boxes = detections['detection_boxes'].numpy()[0]  # Bounding boxes
    classes = detections['detection_classes'].numpy()[0].astype(int)  # Class IDs
    scores = detections['detection_scores'].numpy()[0]  # Confidence scores

    detected_objects = []

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = classes[i]
            label = COCO_LABELS.get(class_id, f"Unknown ({class_id})")
            bbox = boxes[i]

            # Convert bbox to absolute coordinates
            ymin, xmin, ymax, xmax = (bbox * [h, w, h, w]).astype(int)

            # Generate unique labels like "Person 1", "Car 2"
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1
            unique_label = f"{label} {object_counts[label]}"

            # Estimate distance
            distance = estimate_distance((ymin, xmin, ymax, xmax), h)

            detected_objects.append((unique_label, distance, (xmin, ymin, xmax, ymax)))

    return detected_objects
