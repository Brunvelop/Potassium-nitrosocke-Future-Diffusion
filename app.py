import base64
from io import BytesIO

from potassium import Potassium, Request, Response

import torch
from torch import autocast
from diffusers import DiffusionPipeline

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffusionPipeline.from_pretrained(
        "nitrosocke/Future-Diffusion",
        torch_dtype=torch.float16
    ).to(device)
   
    context = {
        "model": model
    }

    return context

def _generate_latent(model, height, width, seed=None, device="cuda"):
    generator = torch.Generator(device=device)

    # Get a new random seed, store it and use it as the generator state
    if not seed:
        seed = generator.seed()
    generator = generator.manual_seed(seed)
    
    image_latent = torch.randn(
        (1, model.unet.in_channels, height // 8, width // 8),
        generator = generator,
        device = device
    )
    return image_latent.type(torch.float16)

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    
    latent = _generate_latent(model, 64*6, 64*6)
    with autocast("cuda"):
        images = model(
            prompt = "future style "+ request.json.get("prompt", None) +" cinematic lights, trending on artstation, avengers endgame, emotional",
            height=64*6,
            width=64*6,
            num_inference_steps = 20,
            guidance_scale = 7.5,
            negative_prompt="duplicate heads bad anatomy extra legs text",
            num_images_per_prompt = 1,
            return_dict=False,
            latents = latent
        )
    image = images[0][0]
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = str(base64.b64encode(buffered.getvalue()))[2:-1]

    return Response(
        json = {"image_base64": image_base64}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()