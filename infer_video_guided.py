import PIL
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import copy
import time
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, DDIMScheduler
from WildVideoDataSet import WildVideoDataSet, CPDataLoader
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
from mmengine import Config
from autoencoder_kl_emasc import AutoencoderKL_EMASC
from utils import remove_overlap,visualize_segmap
from unet_emasc import UNet_EMASC
from video_models.unet import UNet3DConditionModel
from video_models.video_pipeline_guided import VideoPipeline
from einops import rearrange
import imageio
from transformers import CLIPFeatureExtractor, CLIPModel

# seed 
seed = 4
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

config = Config.fromfile('wild_config.py')
opt = copy.deepcopy(config)
# dataset
dataset = WildVideoDataSet(Config.fromfile('wild_config.py'),begin_name='0001.jpg',clothes_name='01486_00.jpg')
opt.datasetting = 'paired'

def collate_fn(examples):
    pcm = torch.stack([example["pcm"] for example in examples])
    parse_cloth = torch.stack([example["parse_cloth"] for example in examples])
    image = torch.stack([example["image"] for example in examples])
    cloth = torch.stack([example["cloth"][opt.datasetting] for example in examples])
    agnostic = torch.stack([example["agnostic"] for example in examples])
    pose = torch.stack([example["pose"] for example in examples])
    c_name = [example['c_name']['paired'] for example in examples]
    im_name = [example['im_name'] for example in examples]
    dino_fea = torch.stack([example["dino_fea"] for example in examples])
    high_frequency_map = torch.stack([example["high_frequency_map"][opt.datasetting] for example in examples])
    parse_other = torch.stack([example["parse_other"] for example in examples])
    parse_upper_mask = torch.stack([example["parse_upper_mask"] for example in examples])
    return {
        "parse_cloth":parse_cloth,
        "pcm": pcm,
        "image": image,
        "agnostic":agnostic,
        "pose":pose,
        "cloth":cloth,
        'c_name':c_name,
        'im_name':im_name,
        "dino_fea":dino_fea,
        "high_frequency_map":high_frequency_map,
        "parse_other":parse_other,
        "parse_upper_mask":parse_upper_mask
    }

batch_size = 16
test_dataloader = CPDataLoader(batch_size=batch_size,workers=4,shuffle=False,dataset=dataset,collate_fn=collate_fn)

if config.unet_path is not None:
    unet = UNet3DConditionModel.from_pretrained_2d(
    config.unet_path, subfolder="unet").to("cuda")
unet.to(dtype=torch.float16)
if config.vae_path is not None:
    vae= AutoencoderKL_EMASC.from_pretrained(
        config.vae_path, subfolder="vae",torch_dtype=torch.float16,low_cpu_mem_usage=False
        ).to("cuda")

ori_vae = AutoencoderKL.from_pretrained(config.model_path, subfolder="vae",torch_dtype=torch.float16).to("cuda")

feature_extractor = CLIPFeatureExtractor.from_pretrained(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
)
clip_model = CLIPModel.from_pretrained(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16
)

scheduler = DDIMScheduler.from_pretrained(config.model_path, subfolder='scheduler')
pipe = VideoPipeline(vae=vae, unet=unet, scheduler=scheduler, clip_model=clip_model, feature_extractor=feature_extractor, ori_vae=ori_vae)
# pipe.enable_vae_sclicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")
pipe._execution_device = torch.device("cuda")

generator = torch.Generator("cuda").manual_seed(seed)

# infer
out_dir = config.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

num_inference_steps = 50
image_guidance_scale = 1
masked_image_guidance_scale = 1
weight_dtype = torch.float16

image_idx = 0
for i in range(1):
    batch = test_dataloader.next_batch()
    # image = batch['image'].cuda()
    c_name = batch['c_name'][0].split('/')[-1]
    im_name = batch['im_name'][0].split('/')[-1]
    c_paired = batch['cloth'].to(device='cuda')

    # condition
    agnostic = batch['agnostic'].to(device='cuda',dtype=weight_dtype)
    pose = batch['pose'].to(device='cuda', dtype=weight_dtype)
    high_frequency_map = batch['high_frequency_map'].to(device='cuda',dtype=weight_dtype)

    # dino_fea
    dino_fea = batch['dino_fea'].to(device='cuda',dtype=weight_dtype)

    # vae decoder
    pcm = batch['pcm'].to(device='cuda', dtype=weight_dtype)
    parse_other = batch['parse_other'].to(device='cuda', dtype=weight_dtype)
    parse_upper_mask = batch['parse_upper_mask'].to(device='cuda', dtype=weight_dtype)
    target_image = batch['image']
    # reshape
    edited_images = pipe(
        masked_image=agnostic,
        # masked_image = parse_other,
        num_inference_steps=num_inference_steps, 
        image_guidance_scale=image_guidance_scale, 
        masked_image_guidance_scale=masked_image_guidance_scale,
        generator=generator,
        pose=pose,
        # cloth_agnostic=parse_other,
        cloth_agnostic=agnostic,
        mask = parse_upper_mask,
        # mask = pcm,
        gt = target_image.to(device='cuda', dtype=weight_dtype),
        high_frequency_map = high_frequency_map,
        dino_fea = dino_fea,
        cloth = c_paired,
    ).images

    outputs = []
    print(c_name)
    for idx, edited_image in enumerate(edited_images):
        if True:
            edited_image = torch.tensor(np.array(edited_image)).permute(2,0,1) / 255.0
            grid = make_image_grid([(c_paired[idx].cpu() / 2 + 0.5),(high_frequency_map[idx].cpu().detach() / 2 + 0.5), (parse_other[idx].cpu().detach() / 2 + 0.5),
            (pose[idx].cpu().detach() / 2 + 0.5),(agnostic[idx].cpu().detach() / 2 + 0.5),
            (target_image[idx].cpu() /2 +0.5), edited_image.cpu(),
            ], nrow=4)
            x = grid.permute(1,2,0)
            x = (x * 255).numpy().astype(np.uint8)
            save_image(grid, os.path.join(out_dir, ('%d.jpg'%image_idx).zfill(6)))
            outputs.append(x)
            image_idx +=1
        else:
            edited_image.save(os.path.join(out_dir, ('%d.jpg'%image_idx).zfill(6)))
            edited_image = np.array(edited_image).astype(np.uint8)
            outputs.append(edited_image)
            image_idx +=1
    imageio.mimsave(os.path.join(out_dir, c_name[:-4]+'.gif'), outputs, 'GIF', duration=500)