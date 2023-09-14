import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from dataclasses import dataclass

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    BaseOutput
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch.nn.functional as F
from .unet import UNet3DConditionModel
from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class VideoPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]

class VideoPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
    
    @torch.no_grad()
    def __call__(
        self,
        masked_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        num_inference_steps: int = 100,
        image_guidance_scale: float = 7.5,
        masked_image_guidance_scale: float = 1.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        pose: Optional[torch.FloatTensor] = None,
        cloth_agnostic: Optional[torch.FloatTensor] = None,
        gt: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        high_frequency_map: Optional[torch.FloatTensor] = None,
        dino_fea: Optional[torch.FloatTensor] = None,
        video_length: Optional[int] = 1,
    ):

        # shape
        # masked_image: (b f) c h w
        # high_frequency_map: (b f) c h w
        # cloth_agnostic: (b f) c h w
        # pose: (b f) c h w
        # mask: (b f) c h w

        # 0. Check inputs
        # self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 1. Define call parameters
        batch_size = 1
        video_length = masked_image.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = image_guidance_scale > 1.0 and masked_image_guidance_scale >= 1.0
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        prompt_embeds = dino_fea.to(device=device, dtype=torch.float16)

        # 3. Preprocess image
        height, width = masked_image.shape[-2:]

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
        # masked_image = rearrange(masked_image, "b c f h w -> (b f) c h w")
        # high_frequency_map = rearrange(high_frequency_map, "b c f h w -> (b f) c h w")
        masked_image_latents, high_frequency_map_latents= self.prepare_image_latents(
            masked_image,
            high_frequency_map,
            prompt_embeds.dtype,
            device,
            generator,
        )
        masked_image_latents = rearrange(masked_image_latents, "(b f) c h w -> b c f h w", f=video_length)
        high_frequency_map_latents = rearrange(high_frequency_map_latents, "(b f) c h w -> b c f h w", f=video_length)
        
        # pose = rearrange(pose, "b c f h w -> (b f) c h w")
        pose_embeds = F.interpolate(pose, scale_factor=(0.125,0.125))
        pose_embeds = pose_embeds.to(device=device, dtype=prompt_embeds.dtype)
        pose_embeds = rearrange(pose_embeds, "(b f) c h w -> b c f h w", f=video_length)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            video_length,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents
                pose_embeds_input = pose_embeds
                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                condition_latent_input = torch.cat([high_frequency_map_latents, masked_image_latents, pose_embeds_input],dim=1)

                # predict the noise residual
                noise_pred = self.unet(scaled_latent_model_input, condition_latent_input, t, encoder_hidden_states=prompt_embeds).sample

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero().item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / (-sigma)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 10. Post-processing
        image = self.decode_latents_emasc(latents, cloth_agnostic, mask)
        # # 11. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 12. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return VideoPipelineOutput(images=image)


    def prepare_image_latents(
        self, masked_image, high_frequency_map, dtype, device, generator=None
    ):

        masked_image = masked_image.to(device=device, dtype=dtype)
        high_frequency_map = high_frequency_map.to(device=device,dtype=dtype)

        if isinstance(generator, list):
            masked_image_latents = [self.vae.encode(masked_image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
            masked_image_latents = torch.cat(masked_image_latents, dim=0)
            high_frequency_map_latents = [self.vae.encode(high_frequency_map[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
            high_frequency_map_latents = torch.cat(high_frequency_map_latents, dim=0)
        else:
            masked_image_latents = self.vae.encode(masked_image).latent_dist.mean
            high_frequency_map_latents = self.vae.encode(high_frequency_map).latent_dist.mean

        masked_image_latents = torch.cat([masked_image_latents], dim=0)
        high_frequency_map_latents = torch.cat([high_frequency_map_latents], dim=0)

        return masked_image_latents, high_frequency_map_latents

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, same_frame_noise=True):

        if latents is None:
            if same_frame_noise:
                shape = (batch_size, num_channels_latents, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                latents = latents.repeat(1, 1, video_length, 1, 1)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents_emasc(self,latents,cloth_agnostic,mask):
        _, inter_features = self.vae.encode(cloth_agnostic, return_inter_features=True)
        feature_conv_out, feature_conv_up_3, feature_conv_up_2, feature_conv_up_1 = inter_features
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        image = self.vae.decode(latents, mask, feature_conv_out, feature_conv_up_1, feature_conv_up_2, feature_conv_up_3).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs