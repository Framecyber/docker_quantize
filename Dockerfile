# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /DOCKER_IMAGE

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install --no-cache-dir torch transformers openvino openvino-dev scikit-learn pillow

# Download and prepare CLIP model
RUN git clone https://github.com/openai/CLIP.git
WORKDIR /DOCKER_IMAGE/CLIP
RUN /opt/venv/bin/pip install -r requirements.txt
RUN /opt/venv/bin/pip install .

# Create directory for OpenVINO quantized CLIP model
WORKDIR /DOCKER_IMAGE
RUN mkdir -p models/clip-openvino

# Copy the local model files into the container
COPY models/clip-vit-base-patch32_int8.xml /DOCKER_IMAGE/models/clip-openvino/clip-vit-base-patch32_int8.xml
COPY models/clip-vit-base-patch32_int8.bin /DOCKER_IMAGE/models/clip-openvino/clip-vit-base-patch32_int8.bin

# Copy the Python script to the container
COPY run_models.py /DOCKER_IMAGE/run_models.py

# Set the working directory
WORKDIR /DOCKER_IMAGE

# Command to run the models
CMD ["/opt/venv/bin/python", "run_models.py"]
