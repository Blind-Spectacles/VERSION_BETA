import cv2
import numpy as np
import tensorflow as tf

# COCO Class Labels
COCO_LABELS = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 5: "Airplane",
    6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic Light",
    11: "Fire Hydrant", 13: "Stop Sign", 14: "Parking Meter", 15: "Bench",
    16: "Bird", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep",
    21: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra", 25: "Giraffe",
    27: "Backpack", 28: "Umbrella", 31: "Handbag", 32: "Tie", 33: "Suitcase",
    34: "Frisbee", 35: "Skis", 36: "Snowboard", 37: "Sports Ball", 38: "Kite",
    39: "Baseball Bat", 40: "Baseball Glove", 41: "Skateboard", 42: "Surfboard",
    43: "Tennis Racket", 44: "Bottle", 46: "Wine Glass", 47: "Cup", 48: "Fork",
    49: "Knife", 50: "Spoon", 51: "Bowl", 52: "Banana", 53: "Apple",
    54: "Sandwich", 55: "Orange", 56: "Broccoli", 57: "Carrot", 58: "Hot Dog",
    59: "Pizza", 60: "Donut", 61: "Cake", 62: "Chair", 63: "Couch",
    64: "Potted Plant", 65: "Bed", 67: "Dining Table", 70: "Toilet", 72: "TV",
    73: "Laptop", 74: "Mouse", 75: "Remote", 76: "Keyboard", 77: "Cell Phone",
    78: "Microwave", 79: "Oven", 80: "Toaster", 81: "Sink", 82: "Refrigerator",
    84: "Book", 85: "Clock", 86: "Vase", 87: "Scissors", 88: "Teddy Bear",
    89: "Hair Drier", 90: "Toothbrush"
}

# Load the MobileNet SSD model
model_dir = "ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model"  # Update this path if needed
model = tf.saved_model.load(model_dir)
detect_fn = model.signatures["serving_default"]

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the model on the input tensor
    detections = detect_fn(input_tensor)

    # Extract detection data
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(int)
    scores = detections['detection_scores'].numpy()[0]

    h, w, _ = frame.shape

    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)

            # Get object name from COCO labels
            class_id = classes[i]
            label = COCO_LABELS.get(class_id, f"Unknown ({class_id})")  # Use 'Unknown' if class ID is missing

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Display label with confidence score
            text = f"{label}: {int(scores[i] * 100)}%"
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
