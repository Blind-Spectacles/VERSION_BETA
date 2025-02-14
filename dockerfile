# Use an official Python base image with TensorFlow support
FROM tensorflow/tensorflow:2.8.0

# Set the working directory inside the container
WORKDIR /app

# Copy the API files into the container
COPY . /app

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port (same as in the Flask app)
EXPOSE 5000

# Command to run the Flask API
CMD ["python", "app.py"]
