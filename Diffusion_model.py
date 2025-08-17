import Diffusion_model
from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

# Generate an image from text
prompt = "A futuristic city with flying cars at sunset"
image = pipe(prompt).images[0]
image.show()
