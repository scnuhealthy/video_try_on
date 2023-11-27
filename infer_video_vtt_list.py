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
from video_models.video_pipeline import VideoPipeline
from einops import rearrange
import imageio
import sys
sys.path.append('./mae')
from mae.models_mae import mae_vit_large_patch16
from VTTDataSet import VTTDataSet

def collate_fn(examples):
    pcm = torch.stack([example["pcm"] for example in examples])
    parse_cloth = torch.stack([example["parse_cloth"] for example in examples])
    image = torch.stack([example["image"] for example in examples])
    cloth = torch.stack([example["cloth"][opt.datasetting] for example in examples])
    agnostic = torch.stack([example["agnostic"] for example in examples])
    pose = torch.stack([example["pose"] for example in examples])
    c_name = [example['c_name'] for example in examples]
    # c_name = [example['c_name']['paired'] for example in examples]
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

# seed 
seed = 5
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# dataset
config = Config.fromfile('config.py')
opt = copy.deepcopy(config)
mode = 'test'
if mode == 'test':
    opt.datamode = config.infer_datamode
    opt.data_list = config.infer_data_list
    opt.datasetting = 'paired'   
else:
    opt.datamode = config.train_datamode
    opt.data_list = config.train_data_list
    opt.datasetting = config.train_datasetting

if config.unet_path is not None:
    unet = UNet3DConditionModel.from_pretrained_2d(
    config.unet_path, subfolder="unet").to("cuda")
# if config.unet_path is not None:
#     unet = UNet3DConditionModel.from_pretrained_2d(
#     config.unet_path, subfolder="unet").to("cuda")
unet.to(dtype=torch.float16)
if config.vae_path is not None:
    vae= AutoencoderKL_EMASC.from_pretrained(
        config.vae_path, subfolder="vae",torch_dtype=torch.float16,low_cpu_mem_usage=False
        ).to("cuda")

scheduler = DDIMScheduler.from_pretrained(config.model_path, subfolder='scheduler')
pipe = VideoPipeline(vae=vae, unet=unet, scheduler=scheduler)
# pipe.enable_vae_sclicing()
pipe.enable_xformers_memory_efficient_attention()
# pipe.to("cuda")
pipe._execution_device = torch.device("cuda")

generator = torch.Generator("cuda").manual_seed(seed)

#     'clothes_person/img/ES121D0RY/ES121D0RY-C11@10.2=cloth_front.jpg', # 2
#     'clothes_person/img/CU721E01G/CU721E01G-C11@11.1=cloth_front.jpg', # 0

# clothes_names = os.listdir('/root/autodl-tmp/zalando-hd-resized/test/image')
# clothes_names = [
#     'clothes_person/img/GP021D08C/GP021D08C-G11@2=cloth_front.jpg', # 1
#     'clothes_person/img/BE621D09B/BE621D09B-M11@10=cloth_front.jpg', #1 easy
#     'clothes_person/img/ED121D0QR/ED121D0QR-K11@10=cloth_front.jpg', # 1
#     'clothes_person/img/DP521E0RX/DP521E0RX-T11@10=cloth_front.jpg', #1
#     'clothes_person/img/TW421DAA2/TW421DAA2-A11@10.1=cloth_front.jpg', # 1
#     'clothes_person/img/GP021D08C/GP021D08C-G11@2=cloth_front.jpg', 
#     'clothes_person/img/4HE21D00F/4HE21D00F-N11@10=cloth_front.jpg',   # 1
#     'clothes_person/img/AN621D0BW/AN621D0BW-Q11@18.1=cloth_front.jpg', # 2
#     'clothes_person/img/AN621DA76/AN621DA76-I11@10=cloth_front.jpg',   # 1
#     'clothes_person/img/BJ721D03B/BJ721D03B-C11@12=cloth_front.jpg'    
#     ]
# clothes_names = [
#     "clothes_person/img/PC721D06P/PC721D06P-G11@2=cloth_front.jpg",
#     #"clothes_person/img/TO721D0DO/TO721D0DO-J11@11=cloth_front.jpg",
#     # "clothes_person/img/ED121D0QS/ED121D0QS-Q11@13.2=cloth_front.jpg",
#     "clothes_person/img/TO221D0IS/TO221D0IS-K11@10=cloth_front.jpg",
#     #"clothes_person/img/AN621D0CG/AN621D0CG-K11@12=cloth_front.jpg",
#     # "clothes_person/img/TO721D0E2/TO721D0E2-K11@14=cloth_front.jpg"
# ]

# failure case of litter 
clothes_names = [
    "clothes_person/img/MG121D006/MG121D006-A11@12=cloth_front.jpg"
]

video_name = 'failure'
for i in range(len(clothes_names)):
    clothes_name = clothes_names[i]
    clothes_name_p = clothes_name.split('/')[-1][:-4]
    print('predicting %s with garment %s' %(video_name, clothes_name_p))
    opt.out_dir = os.path.join(opt.output_root, video_name, clothes_name_p)
    print('Result will be saved into %s' % opt.out_dir)

    dataset = VTTDataSet(opt, target_c_name=clothes_name)
    opt.datasetting = 'paired'

    batch_size = len(dataset)//16*16
    # batch_size = 24
    print(batch_size)
    iters=1
    is_long = False
    test_dataloader = CPDataLoader(batch_size=batch_size,workers=4,shuffle=False,dataset=dataset,collate_fn=collate_fn)

    # infer
    out_dir = opt.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_inference_steps = 30
    image_guidance_scale = 1
    masked_image_guidance_scale = 1
    weight_dtype = torch.float16

    image_idx = 0
    for i in range(iters):
        batch = test_dataloader.next_batch()
        # image = batch['image'].cuda()
        c_name = batch['c_name']
        im_name = batch['im_name']
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
            is_long = is_long
        ).images

        outputs = []
        for idx, edited_image in enumerate(edited_images):
            name1 = c_name[idx].split('/')[-1][:-4] + '+' + im_name[idx].split('/')[-1][:-4] + '.jpg'
            
            # name1 = im_name[idx].split('/')[-1]
            name2 = c_name[idx].split('/')[-1][:-4] + '+' + im_name[idx].split('/')[-1][:-4] + '_cond.jpg'
            
            edited_image.save(os.path.join(out_dir, name1))
            outputs.append(np.array(edited_image).astype(np.uint8))
            edited_image = torch.tensor(np.array(edited_image)).permute(2,0,1) / 255.0
            grid = make_image_grid([(c_paired[idx].cpu() / 2 + 0.5),(high_frequency_map[idx].cpu().detach() / 2 + 0.5), (parse_other[idx].cpu().detach() / 2 + 0.5),
            (pose[idx].cpu().detach() / 2 + 0.5),(agnostic[idx].cpu().detach() / 2 + 0.5),
            (target_image[idx].cpu() /2 +0.5), edited_image.cpu(),
            ], nrow=4)
            # save_image(grid, os.path.join(out_dir, ('%d.jpg'%image_idx).zfill(6)))
            # save_image(grid, os.path.join(out_dir, name2))
            image_idx +=1
            # print(im_name[idx],c_name[idx], name2)
        
        print(len(outputs))
    imageio.mimsave(os.path.join(out_dir, c_name[0].split('/')[-1][:-4]+'.gif'), outputs, 'GIF', duration=50)
        