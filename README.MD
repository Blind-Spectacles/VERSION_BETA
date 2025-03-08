## **🚀 Object Detection & Distance Estimation API**  
**An AI-powered object detection system using MobileNet SSD, with distance estimation for NLP and mobile integration.**  

---

### **📌 Features**
✅ Real-time **object detection** using MobileNet SSD  
✅ **Distance estimation** using bounding box size scaling  
✅ **REST API** for easy integration with NLP & mobile applications  
✅ **Python package** for NLP team  
✅ **TensorFlow Lite (TFLite)** model for mobile apps  
✅ **Dockerized for easy deployment**  

---

# **🛠️ Setup & Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/object-detection.git
cd object-detection
```

---

## **📦 Python Package Setup (For NLP Team)**
### **🔹 Install the Package**
```bash
pip install -e .
```

### **🔹 Usage**
```python
import cv2
from object_detection.detect_objects import detect_objects_and_distance

cap = cv2.VideoCapture(0)
result = detect_objects_and_distance(cap)
cap.release()

print("Detected Objects and Distances:", result)
```
✅ The function returns:
```json
[
    ["Person", 1.2],
    ["Car", 3.5]
]
```

---

## **🌍 REST API Setup (For NLP & Mobile Teams)**
### **🔹 Install Dependencies**
```bash
pip install fastapi uvicorn opencv-python tensorflow numpy
```

### **🔹 Run the API**
```bash
uvicorn app:app --reload
```
✅ API runs on: `http://127.0.0.1:8000`

### **🔹 API Endpoints**
| **Endpoint** | **Method** | **Description** |
|-------------|-----------|----------------|
| `/`         | GET       | Check if API is running |
| `/detect`   | GET       | Detect objects and estimate distances |

### **🔹 Example API Response**
```json
{
    "objects": [
        {"name": "Person", "distance": 1.2},
        {"name": "Car", "distance": 3.5}
    ]
}
```

---

## **📱 Mobile Integration (TensorFlow Lite)**
### **🔹 Convert Model to TFLite**
```bash
python convert_to_tflite.py
```
✅ This creates `mobilenet_ssd.tflite`.

### **🔹 Android Integration (Kotlin)**
1️⃣ Add TensorFlow Lite dependency in `build.gradle`:
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.8.0'
}
```
2️⃣ Load and Run Model in Kotlin:
```kotlin
val interpreter = Interpreter(File("mobilenet_ssd.tflite"))
val input = ByteBuffer.allocateDirect(320 * 320 * 3 * 4)
val output = ByteBuffer.allocateDirect(1000 * 4)

interpreter.run(input, output)
```

---

## **🐳 Docker Deployment**
### **🔹 Build & Run Docker Container**
```bash
docker build -t object_detection .
docker run -p 8000:8000 object_detection
```
✅ The API is now available at `http://localhost:8000/detect`.

---
## **🐳 Docker PULL**
```bash
docker pull franzkingstein/blindglassbeta
```

---

## **👥 Contributors**
- **FRANZ KINGSTEIN N** - AI & Computer Vision Engineer  
- **JOHN JUSVIN** - AI & Computer Vision Engineer  

---

