import cv2

# Initialize MultiTracker
trackers = cv2.legacy.MultiTracker_create()

def initialize_trackers(frame, objects):
    """Initialize KCF trackers for detected objects."""
    global trackers
    trackers = cv2.legacy.MultiTracker_create()
    
    for label, _, (xmin, ymin, xmax, ymax) in objects:
        tracker = cv2.legacy.TrackerKCF_create()
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        trackers.add(tracker, frame, bbox)

def track_objects(frame):
    """Track objects using KCF tracker."""
    success, boxes = trackers.update(frame)
    results = []

    for i, newbox in enumerate(boxes):
        x, y, w, h = [int(v) for v in newbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Object {i+1}"  # Label placeholder
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        results.append((label, (x, y, x + w, y + h)))
    
    return results, frame
