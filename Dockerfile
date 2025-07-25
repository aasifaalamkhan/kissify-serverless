FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Symlinks for Python & pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Pre-download HuggingFace models into cache
RUN python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('SG161222/Realistic_Vision_V5.1_noVAE', torch_dtype='float16')" \
 && python -c \"import requests; r = requests.get('https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3/resolve/main/mm_sd_v15_v2.safetensors'); open('/tmp/motion_adapter.safetensors', 'wb').write(r.content)\" \
 && python -c "from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor; \
               CLIPVisionModelWithProjection.from_pretrained('h94/IP-Adapter', subfolder='models/image_encoder'); \
               CLIPImageProcessor.from_pretrained('h94/IP-Adapter', subfolder='models/image_encoder')"

# Run the handler
CMD ["python", "handler.py"]
