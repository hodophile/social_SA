# VideoPrism Video Understanding – Docker Space (following official Colab)
FROM python: python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps including VideoPrism
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install VideoPrism repository
RUN git clone --depth 1 https://github.com/google-deepmind/videoprism.git videoprism_repo && \
    cd videoprism_repo && \
    # Remove tensorflow line if present to avoid conflicts
    if grep -q tensorflow requirements.txt; then \
        sed -i '/tensorflow/d' requirements.txt; \
    fi && \
    pip install -e . && \
    cd .. && \
    # Install additional deps needed dependencies for the demo
    pip install --no-cache-dir mediapy jax

# Copy application
COPY app.py .

# Hugging Face Spaces expose this port
EXPOSE 7860

# Run Gradio app
CMD ["python", "app.py"]