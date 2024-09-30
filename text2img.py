import torch
from diffusers import StableDiffusionPipline

rand_seed = 42
NUMBER_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 0.5
HEIGHT = 512
WIDTH = 512

model_list=['nota-ai/bk-sdm-small',
            'CompVis/stable-diffusion-v1-4',
            'stabilityai/sdxl-turbo']

def create_pipeline(model_name, device):
    pipeline= StableDiffusionPipline(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    return pipeline

def generate(model_name=model_list[0], device='cpu'):
    if torch.cuda.is_available():
        pipeline=create_pipeline(model_name, device='cuda')
        print('Using CUDA')
    else:
        pipeline=create_pipeline(model_name, device)
        print('Using CPU')
    return pipeline

def text2img(prompt, pipeline):
    images=pipeline(
        prompt,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUMBER_INFERENCE_STEPS,
        generator=rand_seed,
        num_images_per_request=1,
        height=HEIGHT, 
        width=WIDTH
    ).images
    return images[0]