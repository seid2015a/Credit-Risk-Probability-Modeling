# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to save space
# Using --upgrade pip to ensure latest pip
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Set environment variable for MLflow tracking URI (important for container)
# This should match the URI set in src/train.py and src/api/main.py
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 
# For Docker Compose, 'host.docker.internal' refers to the host machine.
# If MLflow server is in another container, use its service name (e.g., http://mlflow_server:5000)

# Command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
