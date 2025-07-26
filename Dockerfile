FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install dependencies and ensure Python 3.10 and pip are set up
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl libgl1 libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure the correct Python version is used
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA 12.1 support (for GPU acceleration)
RUN pip install torch==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Set working directory
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Cache models with CUDA support to avoid re-downloading on each build
RUN python -c "from diffusers import AnimateDiffPipeline; \
    pipeline = AnimateDiffPipeline.from_pretrained('SG161222/Realistic_Vision_V5.1_noVAE', torch_dtype='float16', revision='main')"

RUN python -c "from diffusers import MotionAdapter; \
    adapter = MotionAdapter.from_pretrained('guoyww/animatediff-motion-adapter-v1-5-3')"

RUN python -c "from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor; \
    model = CLIPVisionModelWithProjection.from_pretrained('h94/IP-Adapter', subfolder='models/image_encoder'); \
    processor = CLIPImageProcessor.from_pretrained('h94/IP-Adapter', subfolder='models/image_encoder')"

# Default command to run the app (adjust based on your entry point)
CMD ["python", "handler.py"]
