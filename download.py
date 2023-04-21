import torch
from diffusers import DiffusionPipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffusionPipeline.from_pretrained(
        "nitrosocke/Future-Diffusion",
        torch_dtype=torch.float16
    ).to(device)


if __name__ == "__main__":
    download_model()