import torch
from diffusers import StableDiffusionPipeline

rand_seed = torch.manual_seed(42)
NUMBER_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 0.5
HEIGHT = 512
WIDTH = 512

model_list=['nota-ai/bk-sdm-small',
            'CompVis/stable-diffusion-v1-4',
            'stabilityai/sdxl-turbo',
            'stabilityai/stable-diffusion-xl-base-1.0']

def create_pipeline(model_name, dtype, device):
    pipeline= StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        use_safetensors=True
    ).to(device)
    return pipeline

def generate(model_name=model_list[1], device='cpu'):
    if torch.cuda.is_available():
        pipeline=create_pipeline(model_name, torch.float16, device='cuda')
        print('Using CUDA')
    else:
        pipeline=create_pipeline(model_name, torch.float32, device)
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