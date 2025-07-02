FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./

# Install missing shared libraries for cv2
RUN apt-get update \
 && apt-get install -y libgl1 libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
