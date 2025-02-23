import os
import tensorflow as tf
import cv2
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the COCO labels
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light"
}

# Load the trained object detection model
MODEL_DIR = "C:/Users/Smile/Downloads/VERSION_BETA-final-model/VERSION_BETA-final-model/saved_model"
model = tf.saved_model.load(MODEL_DIR)
detect_fn = model.signatures["serving_default"]

def estimate_distance(box, frame_height):
    """Estimate distance based on bounding box size."""
    ymin, xmin, ymax, xmax = box
    box_height = ymax - ymin
    distance = round(1 / (box_height + 1e-6) * 5, 2)  # Scale factor for approximation
    return distance

def detect_objects_and_distance(camera_input):
    """Detect objects and estimate their distance."""
    ret, frame = camera_input.read()
    if not ret:
        print("Error: Could not read a frame from the camera.")
        return [], frame  # Return empty list and frame
    
    h, w, _ = frame.shape
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(int)
    scores = detections['detection_scores'].numpy()[0]

    results = []
    sentences = []  # List to store NLP sentences
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            label = COCO_LABELS.get(classes[i], f"Unknown ({classes[i]})")
            distance = estimate_distance(boxes[i], h)
            results.append((label, distance))
            sentences.append(f"There is a {label} at approximately {distance} meters.")  # Sentence format for NLP output

    # Return only the sentence output
    return sentences, frame, boxes, classes, scores

def main():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        sentences, frame, boxes, classes, scores = detect_objects_and_distance(cap)

        # Log the NLP sentence output
        if sentences:
            sentence_output = " ".join(sentences)
            print("Sentence Output:", sentence_output)  # Log the NLP output
        else:
            print("Sentence Output: No objects detected.")  # In case no objects are detected

        # Draw boxes on the frame
        h, w, _ = frame.shape
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]
                ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
                label = COCO_LABELS.get(classes[i], f"Unknown ({classes[i]})")
                distance = estimate_distance(boxes[i], h)

                # Draw the bounding box and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                text = f"{label}: {distance}m"
                cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the live camera feed
        cv2.imshow("Object Detection & Distance Estimation", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
