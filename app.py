from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import base64
from object_detection.detect_objects import detect_objects_and_distance
from object_detection.track_object import initialize_trackers, track_objects

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_base64: str

# Dictionary to store detected objects and update their distances
tracked_objects = {}

def convert_np(obj):
    """Convert numpy objects to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.get("/test")
async def test_api():
    """Test endpoint to check server status."""
    return {"message": "Server is running!"}

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """Receives an image file, processes it, and returns detections."""
    image_bytes = await file.read()
    return process_image(image_bytes)

@app.post("/detect")
async def detect(image: ImageRequest):
    """Receives an image as a base64 string, processes it, and returns detections."""
    try:
        image_bytes = base64.b64decode(image.image_base64)
        return process_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")

def process_image(image_bytes: bytes):
    """Converts image bytes to numpy array, detects objects, and returns JSON response."""
    global tracked_objects
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(content={"error": "Invalid image data"}, status_code=400)

    detected_objects = detect_objects_and_distance(frame)
    if not detected_objects:
        return {"message": "No objects detected"}

    json_output = {"detected_objects": []}
    
    for label, distance, (x1, y1, x2, y2) in detected_objects:
        obj_id = f"{label}-{x1}-{y1}"  # Unique key for each object
        if obj_id in tracked_objects:
            tracked_objects[obj_id]["distance"] = convert_np(distance)  # Update distance
        else:
            tracked_objects[obj_id] = {
                "label": label,
                "distance": convert_np(distance),
                "bounding_box": [convert_np(x1), convert_np(y1), convert_np(x2), convert_np(y2)],
                "sentence": f"A {label} was detected at {distance:.2f} meters."
            }

    json_output["detected_objects"] = list(tracked_objects.values())
    
    trackers = initialize_trackers(frame, detected_objects)
    if trackers:
        tracked_objs, _ = track_objects(frame, trackers)
        json_output["tracked_objects"] = []
        for label, (x1, y1, x2, y2) in tracked_objs:
            json_output["tracked_objects"].append({
                "label": label,
                "bounding_box": [convert_np(x1), convert_np(y1), convert_np(x2), convert_np(y2)]
            })
    
    return json_output

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
