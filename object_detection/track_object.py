import cv2

# Check OpenCV version
cv_version = cv2.__version__
print(f"Using OpenCV Version: {cv_version}")

# Initialize MultiTracker
trackers = cv2.legacy.MultiTracker_create()

def initialize_trackers(frame, detected_objects):
    """Initialize CSRT trackers for detected objects."""
    global trackers
    trackers = cv2.legacy.MultiTracker_create()  # Ensure fresh instance

    for _, _, (xmin, ymin, xmax, ymax) in detected_objects:
        tracker = cv2.legacy.TrackerCSRT_create()  # Use CSRT tracker
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)  # Convert to (x, y, width, height)
        trackers.add(tracker, frame, bbox)

def track_objects(frame):
    """Track objects using CSRT tracker."""
    success, boxes = trackers.update(frame)
    
    if not success:
        print("Tracking failed!")
        return [], frame

    results = []
    for i, newbox in enumerate(boxes):
        x, y, w, h = [int(v) for v in newbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Object {i+1}"  # Placeholder label
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        results.append((x, y, x + w, y + h))  # Store the updated bounding box

    return results, frame

# Example usage
if __name__ == "__main__":
    # Load a sample video or image
    video = cv2.VideoCapture(0)  # Use webcam or replace with video file path

    # Detect objects (replace with your object detection logic)
    ret, frame = video.read()
    detected_objects = [(0, 0, (100, 100, 200, 200))]  # Example bounding box (xmin, ymin, xmax, ymax)

    # Initialize trackers
    initialize_trackers(frame, detected_objects)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Track objects
        results, frame = track_objects(frame)

        # Display the frame
        cv2.imshow("Tracking", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
