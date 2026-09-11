# PLM Video Understanding – Docker Space
FROM python:3.12-slim

# Install system deps (ffmpeg for video decoding, build tools for torchvision)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify torchvision is installed
RUN python -c "import torchvision; print('torchvision:', torchvision.__version__)"

# Copy application
COPY app.py .

# Hugging Face Spaces expose this port
EXPOSE 7860

# Run Gradio app bound to 0.0.0.0:7860
CMD ["python", "app.py"]
