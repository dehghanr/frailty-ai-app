# Use slim base with python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies for opencv, mediapipe, etc.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run app with gunicorn and timeout
CMD exec gunicorn --bind 0.0.0.0:8080 --timeout 300 app:app
