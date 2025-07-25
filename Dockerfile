FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Link python3.10 and pip3 to default python/pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the serverless handler
CMD ["python", "handler.py"]
