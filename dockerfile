# Use an official TensorFlow base image
FROM tensorflow/tensorflow:2.8.0

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the API port (same as in the Flask app)
EXPOSE 5000

# Set environment variables (optional, if needed)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask API
CMD ["python", "app.py"]
