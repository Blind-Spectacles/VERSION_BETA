# Distance Estimation for Object Detection

## Introduction
In this document, we outline the method used to detect objects and estimate their distances using a deep learning-based object detection model. The system is built using TensorFlow’s MobileNet SSD model and OpenCV for video processing. The distance estimation technique provides an approximate measure of how far detected objects are from the camera, which can be useful for various applications such as autonomous navigation, surveillance, and robotics.

## Object Detection Model
The model used for object detection is MobileNet SSD (Single Shot MultiBox Detector), which is pre-trained on the COCO (Common Objects in Context) dataset. This model is loaded from a saved directory and used to perform real-time object detection on video frames.

### COCO Class Labels
The COCO dataset contains a set of predefined object classes. Our system currently supports the following classes:

- Person
- Bicycle
- Car
- Motorcycle
- Airplane
- Bus
- Train
- Truck
- Boat
- Traffic Light
- Fire Hydrant
- Stop Sign
- Parking Meter
- Bench
- Bird
- Cat
- Dog
- Horse
- Sheep
- Cow
- Elephant
- Bear
- Zebra
- Giraffe

## Distance Estimation Method
The estimated distance of an object is calculated using its bounding box information obtained from the object detection model. The methodology involves:

1. Extracting the bounding box coordinates of detected objects.
2. Measuring the relative height of the object in the image.
3. Applying a scaling factor to estimate the actual distance.

### Distance Estimation Formula
Given the bounding box coordinates `(ymin, xmin, ymax, xmax)`, the height of the detected object in the image is calculated as:

\[ \text{Object Height} = ymax - ymin \]

To estimate the distance, we use the following empirical formula:

\[ \text{Estimated Distance} = \frac{2.0}{\text{Object Height}} \]

A correction factor is applied to normalize the estimation:

\[ \text{Corrected Distance} = \text{Estimated Distance} \times \left(\frac{0.30}{2.60}\right) \]

This formula is based on a fixed scaling factor derived from observed distances in real-world scenarios.

## Implementation
### Loading the Model
The MobileNet SSD model is loaded using TensorFlow’s SavedModel format:

```python
import tensorflow as tf
MODEL_DIR = "saved_model"
model = tf.saved_model.load(MODEL_DIR)
detect_fn = model.signatures["serving_default"]
```

### Processing Video Input
We capture video frames using OpenCV and convert them into tensors for inference:

```python
import cv2
import numpy as np

def detect_objects_and_distance(camera_input: cv2.VideoCapture):
    ret, frame = camera_input.read()
    if not ret:
        return []
    
    h, w, _ = frame.shape
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    detections = detect_fn(input_tensor)
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(int)
    scores = detections['detection_scores'].numpy()[0]
    
    results = []
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            label = COCO_LABELS.get(classes[i], f"Unknown ({classes[i]})")
            distance = estimate_distance(boxes[i], h)
            results.append((label, distance))
    
    return results
```

### Estimating Distance
The `estimate_distance` function calculates the approximate distance:

```python
def estimate_distance(bbox, image_height):
    ymin, xmin, ymax, xmax = bbox
    object_height = ymax - ymin
    estimated_distance = max(0.1, min(10, 2.0 / object_height))
    corrected_distance = estimated_distance * (0.30 / 2.60)
    return round(corrected_distance, 2)
```

## Conclusion
This document describes the methodology for object detection and distance estimation using deep learning and computer vision techniques. The system estimates distances based on bounding box sizes and applies an empirical correction factor to improve accuracy. This implementation can be extended to various applications, including autonomous vehicles, security surveillance, and assistive technology.

