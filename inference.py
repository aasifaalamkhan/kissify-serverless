import os
import torch
import tempfile
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video
from utils import load_face_images, prepare_ip_adapter_inputs, upload_to_catbox

# ========= GUI status callback support =========
status_callback = None

def set_status_callback(cb):
    global status_callback
    status_callback = cb

def status(msg):
    if status_callback:
        status_callback(msg)

# ========= Patch tqdm globally for GUI progress =========
from tqdm.auto import tqdm as original_tqdm
from tqdm import auto as tqdm_auto

class GuiTqdm(original_tqdm):
    def __init__(self, *args, **kwargs):
        self._callback = kwargs.pop("callback", None)
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        if self._callback and self.total:
            percent = int(self.n / self.total * 100)
            self._callback(f"🧪 Progress: {percent}% ({self.n}/{self.total})")

# Globally patch tqdm
tqdm_auto.tqdm = lambda *args, **kwargs: GuiTqdm(*args, callback=status_callback, **kwargs)

# ========= Load Models =========
base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter"

print("[INFO] Loading models...")

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    ip_adapter_repo_id, subfolder="models/image_encoder"
).to("cuda").eval()

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

motion_adapter = MotionAdapter.from_pretrained(motion_module_id).to("cuda")

pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=motion_adapter,
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
pipe.enable_model_cpu_offload()

pipe.load_ip_adapter(
    ip_adapter_repo_id,
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)
pipe.set_ip_adapter_scale(1.0)

print("[INFO] Pipeline ready.")

# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    from diffusers.utils import logging
    logging.set_verbosity_error()

    status("🧠 Loading and preparing face images...")
    face_images = load_face_images([
        input_data['face_image1'],
        input_data['face_image2']
    ])
    face_images = prepare_ip_adapter_inputs(face_images)

    status("🔍 Encoding faces with IP-Adapter...")
    face_embeds = []
    for face in face_images:
        inputs = image_processor(face, return_tensors="pt", do_rescale=False).to("cuda")
        embeds = image_encoder(**inputs).image_embeds
        face_embeds.append(embeds)

    stacked_embeds = torch.cat(face_embeds, dim=0).mean(dim=0, keepdim=True)

    # Inject face embeddings into UNet
    old_forward = pipe.unet.forward
    def patched_forward(*args, **kwargs):
        if kwargs.get("added_cond_kwargs") is None:
            kwargs["added_cond_kwargs"] = {}
        kwargs["added_cond_kwargs"]["image_embeds"] = stacked_embeds
        return old_forward(*args, **kwargs)
    pipe.unet.forward = patched_forward

    prompt = (input_data.get("prompt") or "").strip()
    if not prompt:
        prompt = "romantic kiss, closeup, cinematic, photorealistic, 4k, trending on artstation"

    status("🎨 Generating animation...")

    result = pipe(
        prompt=prompt,
        num_frames=12,
        guidance_scale=5.0,
        num_inference_steps=10
    )

    video_frames = result.frames[0]

    status("💾 Exporting video...")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
     temp_path = f.name

    export_to_video(video_frames, temp_path, fps=6)

    # ✅ Ensure file is flushed and closed before uploading
    if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
      raise RuntimeError("Exported video file is empty or missing.")

    status("☁️ Uploading to Catbox...")
    video_url = upload_to_catbox(temp_path)

    status("✅ Done!")
    return {"video_url": video_url}
