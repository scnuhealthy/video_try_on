from diffusers import AutoencoderKL
import diffusers
# from diffusers.models.vae import Decoder
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
from diffusers.utils import apply_forward_hook
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Encoder(diffusers.models.vae.Encoder):
    def forward(self, x):
        inter_features = []
        sample = x
        sample = self.conv_in(sample)
        inter_features.append(sample)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            for i, down_block in enumerate(self.down_blocks):
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                if i > 0:
                    inter_features.append(sample)
            # middle
            sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for i, down_block in enumerate(self.down_blocks):
                sample = down_block(sample)
                if i < 3:
                    inter_features.append(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample, inter_features


class Decoder(diffusers.models.vae.Decoder):
    def __init__(self,
            in_channels=3,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
                 ):
        super().__init__(in_channels,out_channels,up_block_types,block_out_channels,layers_per_block,norm_num_groups,act_fn)
        self.emasc_conv_up_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(512, 512,kernel_size=3, padding=1, stride=1)),
        )
        self.emasc_conv_up_2 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, 512,kernel_size=3, padding=1, stride=1)),
        )
        self.emasc_conv_up_3 = nn.Sequential(
            nn.Conv2d(128, 128,kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(128, 256,kernel_size=3, padding=1, stride=1)),
        )

        self.emasc_conv_out = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(128, 128,kernel_size=3, padding=1, stride=1)),
        )      

    def forward(self, z, mask, feature_conv_out, feature_conv_up_1, feature_conv_up_2, feature_conv_up_3):
        sample = z
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # middle
            sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)
            sample = sample.to(upscale_dtype)

            # up
            for i, up_block in enumerate(self.up_blocks):
                if i == 1:
                    p = torch.utils.checkpoint.checkpoint(create_custom_forward(self.emasc_conv_up_1), feature_conv_up_1)
                    sample = sample + p * mask
                elif i == 2:
                    p = torch.utils.checkpoint.checkpoint(create_custom_forward(self.emasc_conv_up_2), feature_conv_up_2)
                    sample = sample + p * mask
                elif i == 3:
                    p = torch.utils.checkpoint.checkpoint(create_custom_forward(self.emasc_conv_up_3), feature_conv_up_3)
                    sample = sample + p * mask
                else:
                    pass
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample)
        else:
            # middle
            sample = self.mid_block(sample)
            sample = sample.to(upscale_dtype)

            # up
            for i, up_block in enumerate(self.up_blocks):
                if i == 1:
                    feature_conv_up_1 = nn.functional.interpolate(feature_conv_up_1, scale_factor=(2,2), mode='nearest')
                    resized_mask = nn.functional.interpolate(mask, scale_factor=(0.25,0.25),mode='nearest') < 0.5
                    # print(sample.shape,feature_conv_up_1.shape, resized_mask.shape)
                    sample = sample + self.emasc_conv_up_1(feature_conv_up_1) * resized_mask
                elif i == 2:
                    feature_conv_up_2 = nn.functional.interpolate(feature_conv_up_2, scale_factor=(2,2), mode='nearest')
                    resized_mask = nn.functional.interpolate(mask, scale_factor=(0.5,0.5),mode='nearest') < 0.5
                    # print(sample.shape,feature_conv_up_2.shape, resized_mask.shape)
                    sample = sample + self.emasc_conv_up_2(feature_conv_up_2) * resized_mask
                elif i == 3:
                    feature_conv_up_3 = nn.functional.interpolate(feature_conv_up_3, scale_factor=(2,2), mode='nearest')
                    resized_mask = nn.functional.interpolate(mask, scale_factor=(1.0,1.0),mode='nearest') < 0.5
                    # print(resized_mask.shape, torch.sum(resized_mask))
                    # print(sample.shape,feature_conv_up_3.shape, resized_mask.shape)
                    sample = sample + self.emasc_conv_up_3(feature_conv_up_3) * resized_mask
                else:
                    pass
                sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        p = self.emasc_conv_out(feature_conv_out)
        # print(sample.shape,feature_conv_out.shape, p.shape, mask.shape)
        sample = sample + p * mask
        sample = self.conv_out(sample)

        return sample


class AutoencoderKL_EMASC(AutoencoderKL):
    def __init__(self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
            block_out_channels: Tuple[int] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 4,
            norm_num_groups: int = 32,
            sample_size: int = 32,
            scaling_factor: float = 0.18215
        ):
        super().__init__(in_channels,out_channels,down_block_types,up_block_types,block_out_channels,layers_per_block,
                        act_fn,latent_channels,norm_num_groups,sample_size,scaling_factor)
        
        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

    @apply_forward_hook
    def encode(self, x, return_inter_features=False, return_dict = True):
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        h, inter_features = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        if return_inter_features:
            return AutoencoderKLOutput(latent_dist=posterior), inter_features
        else:
            return AutoencoderKLOutput(latent_dist=posterior)


    def _decode(self, z, mask, feature_conv_out, feature_conv_up_1, feature_conv_up_2, feature_conv_up_3, return_dict = True):
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z, mask, feature_conv_out, feature_conv_up_1, feature_conv_up_2, feature_conv_up_3)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z, mask, feature_conv_out, feature_conv_up_1, feature_conv_up_2, feature_conv_up_3, return_dict = True):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, mask, feature_conv_out, feature_conv_up_1, feature_conv_up_2, feature_conv_up_3).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)



def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def collate_fn(examples):
    pcm = torch.stack([example["pcm"] for example in examples])
    pcm = pcm.to(memory_format=torch.contiguous_format).float()
    image = torch.stack([example["image"] for example in examples])
    image = image.to(memory_format=torch.contiguous_format).float()
    agnostic = torch.stack([example["agnostic"] for example in examples])
    agnostic = agnostic.to(memory_format=torch.contiguous_format).float()
    parse_cloth = torch.stack([example["parse_cloth"] for example in examples])
    parse_cloth = parse_cloth.to(memory_format=torch.contiguous_format).float()
    parse_other = torch.stack([example["parse_other"] for example in examples])
    parse_upper_mask = torch.stack([example["parse_upper_mask"] for example in examples])
    return {
        "pcm": pcm,
        "image": image,
        "agnostic":agnostic,
        "parse_cloth":parse_cloth,
        "parse_other":parse_other,
        "parse_upper_mask":parse_upper_mask
    }

if __name__ == '__main__':
    from mmengine import Config
    import copy
    from CPDataset_HD import CPDatasetTest, CPDataset, CPDataLoader
    from PIL import Image
    import numpy as np
    
    vae_path = 'parse_other_vae_model/checkpoint-2000'
    # vae_path = 'save_models/HR_VITON_vae/'
    # vae_path = '/data1/hzj/clothes_model_vae/'
    # vae_path = 'runwayml/stable-diffusion-v1-5'
    vae= AutoencoderKL_EMASC.from_pretrained(
        vae_path, subfolder="vae",torch_dtype=torch.float32, low_cpu_mem_usage=False
    ).cuda()
    
    config = Config.fromfile('config.py')
    opt = copy.deepcopy(config)
    opt.datamode = config.infer_datamode
    opt.data_list = config.infer_data_list
    opt.datasetting = config.infer_datasetting

    # train_dataset = CPDatasetTest(opt)
    # image = train_dataset[0]['image'].unsqueeze(0).cuda()
    # agnostic = train_dataset[0]['agnostic'].unsqueeze(0).cuda()
    # mask = train_dataset[0]['pcm'].unsqueeze(0).cuda()

    test_dataset = CPDatasetTest(opt)
    test_dataloader = CPDataLoader(batch_size=1,workers=4,shuffle=True,dataset=test_dataset,collate_fn=collate_fn)

    for i in range(15):
        batch = test_dataloader.next_batch()
        image = batch['image'].cuda()
        # mask = batch['pcm'].cuda()
        mask = batch['parse_upper_mask'].cuda()
        agnostic = batch['parse_other'].cuda()
        out = image[0].cpu() /2 +0.5
        out = out.detach().permute(1,2,0).numpy()
        out = (out * 255).astype(np.uint8)
        out = Image.fromarray(out)
        out.save('%d_test_ori.png' % i)

        print(image.shape,mask.shape, torch.mean(mask))

        # latents, inter_features = vae.encode(image)
        latents, _ = vae.encode(image,return_inter_features=True)
        _, inter_features = vae.encode(agnostic,return_inter_features=True)
        latents = latents.latent_dist.sample()
        feature_conv_out, feature_conv_up_3, feature_conv_up_2, feature_conv_up_1 = inter_features
        print(feature_conv_out.shape, feature_conv_up_1.shape, feature_conv_up_2.shape, feature_conv_up_3.shape)
        latents = latents * vae.config.scaling_factor

        latents = 1 / vae.config.scaling_factor * latents
        pred_images = vae.decode(latents, mask, feature_conv_out, feature_conv_up_1, feature_conv_up_2, feature_conv_up_3).sample
        pred_images = pred_images.clamp(-1, 1)
        out = pred_images[0].cpu() /2 +0.5
        out = out.detach().permute(1,2,0).numpy()
        out = (out * 255).astype(np.uint8)
        out = Image.fromarray(out)
        out.save('%d_test.png' % i)

# with torch.no_grad():
#     latents = vae.encode(image).latent_dist.sample()
#     latents = latents * vae.config.scaling_factor
#     l2 = copy.deepcopy(latents)
#     latents = decode_latents(vae, latents)
# image = numpy_to_pil(latents)
# image[0].save('test.png')

# latents = 1 / vae.config.scaling_factor * l2
# pred_images = vae.decode(latents).sample
# out = pred_images.cpu() /2 +0.5
# out = out.clamp(0,1)

# # out = out.detach().permute(0,2,3,1).float().numpy()
# out = out[0]
# # out = out.detach().permute(1,2,0).float().numpy()

# # out = (out * 255).astype("uint8")
# # print(out.shape, np.max(out),np.min(out))
# # out = Image.fromarray(out)
# # out.save('test2.png')

# out = out.detach().permute(1,2,0).float().numpy()
# out = (out * 255).astype("uint8")
# print(out.shape,np.min(out),np.max(out))
# out = Image.fromarray(out)
# out.save('test4.png')
# # def decode_latents(vae, latents):
# #     latents = 1 / vae.config.scaling_factor * latents
# #     image = vae.decode(latents).sample
# #     image = (image / 2 + 0.5).clamp(0, 1)
# #     # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
# #     image = image.cpu().permute(0, 2, 3, 1).float().numpy()
# #     return image
