import cv2
from object_detection.detect_objects import detect_objects_and_distance
from object_detection.track_object import initialize_trackers, track_objects

def main():
    """Main function to run object detection and tracking."""
    camera = cv2.VideoCapture(0)  # Open webcam

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    trackers = None  # Store tracker information

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect objects and their distances
        detected_objects = detect_objects_and_distance(frame)

        # Initialize trackers only if objects are detected
        if trackers is None and detected_objects:
            trackers = initialize_trackers(frame, detected_objects)

        # Track objects in subsequent frames
        if trackers is not None:
            tracked_objects, frame = track_objects(frame, trackers)

            # Display tracked objects
            for label, (x1, y1, x2, y2) in tracked_objects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display detected objects with labels
        for label, distance, (x1, y1, x2, y2) in detected_objects:
            print(f"Object: {label}, Distance: {distance}m, Coordinates: ({x1}, {y1}), ({x2}, {y2})")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({distance}m)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow("Object Tracking", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()