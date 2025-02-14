import tensorflow as tf

# Load the trained MobileNet SSD model
MODEL_DIR = "saved_model"
model = tf.saved_model.load(MODEL_DIR)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize for mobile
tflite_model = converter.convert()

# Save the TFLite model
with open("mobilenet_ssd.tflite", "wb") as f:
    f.write(tflite_model)

print(" TFLite model saved as mobilenet_ssd.tflite")
