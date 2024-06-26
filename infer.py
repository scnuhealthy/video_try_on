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
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
from mmengine import Config
from autoencoder_kl_emasc import AutoencoderKL_EMASC
from utils import remove_overlap,visualize_segmap
from unet_emasc import UNet_EMASC
from blended_cloth_pipeline import BlendedClothPipeline
# from new_pipe import BlendedClothPipeline

# seed 
seed = 17
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

mode = 'test'
# config
config = Config.fromfile('config.py')
opt = copy.deepcopy(config)
if mode == 'test':
    opt.datamode = config.infer_datamode
    opt.data_list = config.infer_data_list
    opt.datasetting = config.infer_datasetting   
else:
    opt.datamode = config.train_datamode
    opt.data_list = config.train_data_list
    opt.datasetting = config.train_datasetting

# dataset
def collate_fn(examples):
    pcm = torch.stack([example["pcm"] for example in examples])
    parse_cloth = torch.stack([example["parse_cloth"] for example in examples])
    image = torch.stack([example["image"] for example in examples])
    cloth = torch.stack([example["cloth"][opt.datasetting] for example in examples])
    # cloth_mask = torch.stack([example["cloth_mask"][opt.datasetting] for example in examples])
    agnostic = torch.stack([example["agnostic"] for example in examples])
    # parse_agnostic = torch.stack([example["parse_agnostic"] for example in examples])
    pose = torch.stack([example["pose"] for example in examples])
    # densepose = torch.stack([example["densepose"] for example in examples])
    im_name = [example['im_name'] for example in examples]
    c_name = [example['c_name'][opt.datasetting] for example in examples]
    dino_fea = torch.stack([example["dino_fea"] for example in examples])
    high_frequency_map = torch.stack([example["high_frequency_map"][opt.datasetting] for example in examples])
    # hog_map = torch.stack([example["hog_map"][opt.datasetting] for example in examples])
    parse_other = torch.stack([example["parse_other"] for example in examples])
    parse_upper_mask = torch.stack([example["parse_upper_mask"] for example in examples])
    parse = torch.stack([example["parse"] for example in examples])
    return {
        "parse_cloth":parse_cloth,
        "pcm": pcm,
        "image": image,
        "agnostic":agnostic,
        # "parse_agnostic":parse_agnostic,
        # "densepose":densepose,
        "pose":pose,
        "cloth":cloth,
        # 'cloth_mask':cloth_mask,
        'im_name':im_name,
        "c_name":c_name,
        "dino_fea":dino_fea,
        "high_frequency_map":high_frequency_map,
        # "hog_map":hog_map,
        "parse_other":parse_other,
        "parse_upper_mask":parse_upper_mask,
        'parse':parse
    }

if opt.test_dataset == 'DressCode':
    from DressCodeDataSet import DressCodeDataSet, CPDataLoader
    opt.datasetting = 'paired'
    dataset = DressCodeDataSet(opt)
elif opt.test_dataset == 'VITON':
    from CPDataset_HD import CPDatasetTest, CPDataset, CPDataLoader
    dataset = CPDatasetTest(opt)
elif opt.test_dataset == 'Wild':
    from WildVideoDataSet import WildVideoDataSet,CPDataLoader
    dataset = WildVideoDataSet(Config.fromfile('wild_config.py'))
    opt.datasetting = 'paired'
elif opt.test_dataset == 'VTT':
    from VideoDataSet import VideoDataSet, CPDataLoader
    dataset = VideoDataSet(opt)
    opt.datasetting = 'paired'
elif opt.test_dataset == 'TikTok':
    from TikTokDataSet import TikTokDataSet, CPDataLoader
    dataset = TikTokDataSet(opt)
    opt.datasetting = 'paired'

batch_size = 10
test_dataloader = CPDataLoader(batch_size=batch_size,workers=4,shuffle=True,dataset=dataset,collate_fn=collate_fn)

pipe = BlendedClothPipeline.from_pretrained(config.model_path, safety_checker=None, requires_safety_checker=False,torch_dtype=torch.float16).to("cuda")
if config.unet_path is not None:
    unet = UNet_EMASC.from_pretrained(
    config.unet_path, subfolder="unet",torch_dtype=torch.float16
    ).to("cuda")
    pipe.unet = unet

if config.vae_path is not None:
    vae= AutoencoderKL_EMASC.from_pretrained(
       config.vae_path, subfolder="vae",torch_dtype=torch.float16,low_cpu_mem_usage=False
    ).to("cuda")
    pipe.vae = vae

pipe.scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    )
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
for i in range(0,20):
# for i, batch in enumerate(test_dataloader.data_loader):
    batch = test_dataloader.next_batch()
    # image = batch['image'].cuda()
    im_name = batch['im_name']
    c_name = batch['c_name']
    c_paired = batch['cloth'].to(device='cuda')

    # condition
    agnostic = batch['agnostic'].to(device='cuda',dtype=weight_dtype)
    pose = batch['pose'].to(device='cuda', dtype=weight_dtype)
    high_frequency_map = batch['high_frequency_map'].to(device='cuda',dtype=weight_dtype)
    # high_frequency_map = torch.zeros_like(high_frequency_map).to(device='cuda',dtype=weight_dtype)
    # hog_map = batch['hog_map'].to(device='cuda',dtype=weight_dtype)

    # dino_fea
    dino_fea = batch['dino_fea'].to(device='cuda',dtype=weight_dtype)

    # vae decoder
    pcm = batch['pcm'].to(device='cuda', dtype=weight_dtype)
    parse_other = batch['parse_other'].to(device='cuda', dtype=weight_dtype)
    parse_upper_mask = batch['parse_upper_mask'].to(device='cuda', dtype=weight_dtype)
    target_image = batch['image']

    # vis
    parse = batch['parse']

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
        # mask = parse_upper_mask,
        mask = pcm,
        gt = target_image.to(device='cuda', dtype=weight_dtype),
        high_frequency_map = high_frequency_map,
        # hog_map = hog_map,
        dino_fea = dino_fea,
    ).images

    for idx, edited_image in enumerate(edited_images):
        if opt.test_dataset == 'TikTok':
            name1 = im_name[idx].split('/')[1] + '+' + im_name[idx].split('/')[-1][:-4] + '+' + c_name[idx].split('/')[-1]
            name2 = im_name[idx].split('/')[1] + '+' + im_name[idx].split('/')[-1][:-4] + '+' + c_name[idx].split('/')[-1][:-4] + '_cond.jpg'
        elif opt.test_dataset == 'VITON' or opt.test_dataset == 'DressCode':
            name1 = im_name[idx].split('/')[0] + '+' + c_name[idx]
            name2 = im_name[idx].split('/')[0] + '+' + c_name[idx].split('/')[0][:-4] + '_cond.jpg'
        else:
            pass
        # edited_image.save(os.path.join(out_dir, name1))

        edited_image = torch.tensor(np.array(edited_image)).permute(2,0,1) / 255.0
        grid = make_image_grid([(c_paired[idx].cpu() / 2 + 0.5),(high_frequency_map[idx].cpu().detach() / 2 + 0.5), (parse_other[idx].cpu().detach() / 2 + 0.5),
        (pose[idx].cpu().detach() / 2 + 0.5),(agnostic[idx].cpu().detach() / 2 + 0.5), visualize_segmap(parse[idx].unsqueeze(0).cpu()),
        (target_image[idx].cpu() /2 +0.5), edited_image.cpu(),
        ], nrow=4)
        # save_image(grid, os.path.join(out_dir, ('%d.jpg'%image_idx).zfill(6)))
        save_image(grid, os.path.join(out_dir, name2))
        image_idx +=1
        print(im_name[idx],c_name[idx], name2)

    # for idx, edited_image in enumerate(edited_images):
    #     edited_image = torch.tensor(np.array(edited_image)).permute(2,0,1) / 255.0
    #     grid = make_image_grid([(c_paired[idx].cpu() ),(high_frequency_map[idx].cpu().detach() ), (parse_other[idx].cpu().detach() ),
    #     (pose[idx].cpu().detach() ),(agnostic[idx].cpu().detach() ),
    #     (target_image[idx].cpu() ), edited_image.cpu(),
    #     ], nrow=4)
    #     save_image(grid, os.path.join(out_dir, ('%d.jpg'%image_idx).zfill(6)))
    #     image_idx +=1

