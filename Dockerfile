# Step 5: Install dependencies including PyTorch with CUDA
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.1.0+cu118 diffusers transformers accelerate

# Step 6: Pre-download models
RUN mkdir /app/models && \
    python3 -c "from diffusers import StableDiffusionPipeline; \
    StableDiffusionPipeline.from_pretrained('SG161222/Realistic_Vision_V5.1_noVAE', torch_dtype='float16', cache_dir='/app/models/SG161222/Realistic_Vision_V5.1_noVAE')"

# Copy the rest of your app code
COPY . .

# Expose port
EXPOSE 8000

# Start the app
CMD ["python3", "app.py"]
