import PIL
import numpy as np
import torch
import requests
from PIL import Image
import torchvision.transforms as transforms
from typing import Union
from einops import rearrange
from tqdm import tqdm
import cv2

def get_high_frequency_map(img_path):
    img = cv2.imread(img_path,0)
    rgb_img = np.array(Image.open(img_path))
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = np.abs(sobelx)
    sobely = np.abs(sobely)
    sobel = sobelx  + sobely
    t = (sobelx  + sobely) >  120
    p = np.zeros((rgb_img.shape[0],rgb_img.shape[1],3))
    p[:,:,0] = t * rgb_img[:,:,0]
    p[:,:,1] = t * rgb_img[:,:,1]
    p[:,:,2] = t * rgb_img[:,:,2]
    p = Image.fromarray(p.astype(np.uint8))
    return p

def get_cond_color(cond_image, mask_size=16):
    H, W = cond_image.size
    cond_image = cond_image.resize((W // mask_size, H // mask_size), Image.BICUBIC)
    color = cond_image.resize((H, W), Image.NEAREST)
    return color

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def ndim_tensor2im(image_tensor, imtype=np.uint8, batch=0):
    image_numpy = image_tensor[batch].cpu().float().numpy()
    result = np.argmax(image_numpy, axis=0)
    return result.astype(imtype)

def visualize_segmap(input, multi_channel=True, tensor_out=True, batch=0) :
    palette = [
        0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51,
        254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85,
        85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220,
        0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0
    ]
    input = input.detach()
    if multi_channel :
        input = ndim_tensor2im(input,batch=batch)
    else :
        input = input[batch][0].cpu()
        input = np.asarray(input)
        input = input.astype(np.uint8)
    input = Image.fromarray(input, 'P')
    input.putpalette(palette)

    if tensor_out :
        trans = transforms.ToTensor()
        return trans(input.convert('RGB'))

    return input


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, tokenizer, text_encoder):
    print(text_encoder.device)
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(text_encoder.device))[0]
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(text_encoder.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet, conditions):
    noise_pred = unet(torch.cat([latents, conditions], dim=1), t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(tokenizer, text_encoder, unet, ddim_scheduler, latent, num_inv_steps, prompt, conditions):
    context = init_prompt(prompt, tokenizer, text_encoder)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, unet, conditions)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(tokenizer, text_encoder, unet, ddim_scheduler, video_latent, num_inv_steps, conditions, prompt=""):
    ddim_latents = ddim_loop(tokenizer, text_encoder, unet, ddim_scheduler, video_latent, num_inv_steps, prompt, conditions)
    return ddim_latents