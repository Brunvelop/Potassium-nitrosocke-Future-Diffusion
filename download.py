import torch
from diffusers import DiffusionPipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    DiffusionPipeline.from_pretrained(
        "nitrosocke/Future-Diffusion",
        torch_dtype=torch.float32
    ).to('cuda')

if __name__ == "__main__":
    download_model()