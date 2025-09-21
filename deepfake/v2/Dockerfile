# Use official PyTorch image (with CPU support only).
# If you need CUDA, replace with pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
FROM python:3.13.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, PIL, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#separate layer for uvicorn to not install requirements.txt again
RUN pip install --no-cache-dir uvicorn[standard]

RUN pip install --no-cache-dir opencv-python

RUN pip install --no-cache-dir grad-cam

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
