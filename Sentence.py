import cv2
import json
import numpy as np
import tensorflow as tf
from object_detection.detect_objects import detect_objects_and_distance
from object_detection.track_object import initialize_trackers, track_objects

# Enable TensorFlow GPU acceleration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow is using the GPU:", gpus)
    except RuntimeError as e:
        print(e)

# Check if OpenCV has CUDA support
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    use_cuda = True
    print("Using OpenCV with CUDA acceleration")
else:
    use_cuda = False
    print("CUDA not available for OpenCV, using CPU instead")

def preprocess_frame(frame):
    """Preprocess the frame for object detection using GPU."""
    if use_cuda:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        return gpu_frame
    return frame

def assign_ids(tracked_objects, prev_objects):
    """
    Maintain a consistent object ID by comparing bounding boxes.
    """
    new_objects = {}
    assigned_ids = set()

    for obj in tracked_objects:
        label, (x1, y1, x2, y2) = obj
        min_distance = float("inf")
        best_match = None

        # Find the closest previous object
        for prev_id, (prev_label, (px1, py1, px2, py2)) in prev_objects.items():
            if prev_label == label and prev_id not in assigned_ids:
                # Compute distance between bounding box centers
                prev_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                new_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance = np.linalg.norm(np.array(prev_center) - np.array(new_center))

                if distance < min_distance:
                    min_distance = distance
                    best_match = prev_id

        if best_match is not None:
            new_objects[best_match] = (label, (x1, y1, x2, y2))
            assigned_ids.add(best_match)
        else:
            new_id = f"{label} {len(prev_objects) + 1}"
            new_objects[new_id] = (label, (x1, y1, x2, y2))

    return new_objects

def main():
    """Main function to run object detection and tracking with GPU acceleration."""
    camera = cv2.VideoCapture(0)  # Open webcam

    if not camera.isOpened():
        print(json.dumps({"error": "Could not open webcam"}))
        return

    trackers = None  # Store tracker information
    prev_objects = {}  # Store previous object IDs

    while True:
        ret, frame = camera.read()
        if not ret:
            print(json.dumps({"error": "Failed to capture frame"}))
            break

        frame = preprocess_frame(frame)  # Convert frame to GPU if available

        # Detect objects and their distances
        detected_objects = detect_objects_and_distance(frame)

        # Track objects persistently
        if trackers is None and detected_objects:
            trackers = initialize_trackers(frame, detected_objects)
        if trackers is not None:
            tracked_objects, frame = track_objects(frame, trackers)
            prev_objects = assign_ids(tracked_objects, prev_objects)

        # Generate JSON output with dynamic distances
        json_output = {"detected_objects": []}
        for obj_id, (label, (x1, y1, x2, y2)) in prev_objects.items():
            # Find the corresponding distance for the tracked object
            for d_label, distance, bbox in detected_objects:
                if d_label == label and bbox == (x1, y1, x2, y2):
                    sentence = f"A {obj_id} was detected at a distance of {distance:.2f} meters with bounding box coordinates ({x1}, {y1}) to ({x2}, {y2})."
                    json_output["detected_objects"].append({"sentence": sentence})

        print(json.dumps(json_output, indent=4))  # Print JSON to terminal

        # Draw tracked objects on the frame
        for obj_id, (label, (x1, y1, x2, y2)) in prev_objects.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if use_cuda:
            frame = frame.download()  # Convert back to CPU if CUDA is enabled

        cv2.imshow("Object Tracking", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
