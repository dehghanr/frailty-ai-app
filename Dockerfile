# Install system dependencies for cv2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (using headless OpenCV)
COPY requirements.txt .
RUN sed -i 's/opencv-python/opencv-python-headless/' requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the HTTP port
EXPOSE 8080

# Launch the app using Gunicorn on the correct port
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]


