import logging
import math
import os
from pathlib import Path

import accelerate
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from unet_emasc import UNet_EMASC
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import itertools
from diffusers.models.vae import Encoder
from CPDataset_HD import CPDataset
from VTTDataSet import VTTDataSet
from DressCodeDataSet import DressCodeDataSet
from TikTokDataSet import TikTokDataSet
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
from mmengine import Config
from utils import remove_overlap


logger = get_logger(__name__, log_level="INFO")


def parse_args():

    args = Config.fromfile('config.py')
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    # data
    args.datamode = args.train_datamode
    args.data_list = args.train_data_list
    args.datasetting = args.train_datasetting
    
    return args


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def main():
    args = parse_args()
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    # from accelerate import DistributedDataParallelKwargs
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
        #kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if os.path.join(args.output_dir,'grid') is not None:
            os.makedirs(os.path.join(args.output_dir,'grid'), exist_ok=True)
        args.dump(os.path.join(args.output_dir, 'experiment_config.py'))
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    # unet = UNet_EMASC.from_pretrained(
    #    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision,low_cpu_mem_usage=False
    # )
    unet = UNet_EMASC.from_pretrained(
       'model_VITON_512_fixbug/checkpoint-120000', subfolder="unet", revision=args.non_ema_revision, # low_cpu_mem_usage=False
    )
    # unet = UNet_EMASC(cross_attention_dim=768, block_out_channels=(320,320,640,640),norm_num_groups=32,sample_size=64, in_channels=4, out_channels=4)

    # encoder hidden proj
    # encoder_hid_dim = 1024
    # unet.register_to_config(encoder_hid_dim=encoder_hid_dim)
    # unet.encoder_hid_proj = nn.Linear(encoder_hid_dim, unet.cross_attention_dim)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(train_params, model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            print('xformer_version', xformers_version)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                try:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                except:
                    model_path = os.path.join(output_dir, "warp.pth")
                    torch.save(model.state_dict(), model_path)
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                try:
                    load_model = UNet_EMASC.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                except:
                    pass
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
       unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    for name, p in unet.named_parameters():
        if 'cloth' not in name and 'down' in name and 'fuse' not in name:
            p.requires_grad = False
        if 'conv_in' in name and 'cloth' not in name:
            p.requires_grad = False
    train_params = [p for n,p in unet.named_parameters() if p.requires_grad]
    frozen_params = [p for n,p in unet.named_parameters() if not p.requires_grad]
    # train_params = unet.parameters()
    optimizer = optimizer_cls(
        train_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )  
    
    train_dataset0 = TikTokDataSet(args)
    train_dataset1 = CPDataset(args)
    train_dataset2 = DressCodeDataSet(args)
    weights = [3.0] * len(train_dataset0) + [1.0] * len(train_dataset1) + [1.0] * len(train_dataset2)
    train_dataset = ConcatDataset([train_dataset0, train_dataset1, train_dataset2])
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

    print('Length DataSet', len(train_dataset0), len(train_dataset1), len(train_dataset2))

    def collate_fn(examples):
        image = torch.stack([example["image"] for example in examples])
        cloth = torch.stack([example["cloth"]["paired"] for example in examples])
        # cloth_mask = torch.stack([example["cloth_mask"]["paired"] for example in examples])
        # parse_agnostic = torch.stack([example["parse_agnostic"]for example in examples])
        # densepose = torch.stack([example["densepose"] for example in examples])
        agnostic = torch.stack([example["agnostic"] for example in examples])
        # parse_other = torch.stack([example["parse_other"] for example in examples])
        pose = torch.stack([example["pose"] for example in examples])
        dino_fea = torch.stack([example["dino_fea"] for example in examples])
        high_frequency_map = torch.stack([example["high_frequency_map"]["paired"] for example in examples])
        # color_map = torch.stack([example["color_map"]["paired"] for example in examples])
        return {
            "image":image,
            "cloth":cloth,
            # "cloth_mask":cloth_mask,
            "agnostic":agnostic,
            "pose":pose,
            # "densepose":densepose,
            # "parse_agnostic":parse_agnostic,
            "dino_fea":dino_fea,
            "high_frequency_map":high_frequency_map,
            # "color_map":color_map,
            # 'parse_other':parse_other
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        sampler=sampler,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        # tocg.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            # target 
            image = batch["image"]
            
            # agnostic
            cloth_agnostic = batch["agnostic"]
            # cloth_agnostic = batch["parse_other"]
            pose = batch["pose"]
            high_frequency_map = batch['high_frequency_map']
            # color_map = batch["color_map"]
            # dino_fea
            dino_fea = batch['dino_fea']

            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(image.to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the pose embedding for conditioning.
                encoder_hidden_states = dino_fea

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                high_frequency_map = vae.encode(high_frequency_map.to(weight_dtype)).latent_dist.mode()
                # color_map = vae.encode(color_map.to(weight_dtype)).latent_dist.mode()

                # agnostic encode
                masked_image_embeds = vae.encode(cloth_agnostic.to(weight_dtype)).latent_dist.mode()
                pose_embeds = F.interpolate(pose.to(weight_dtype), scale_factor=(0.125,0.125))

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                condition_latent = torch.cat([high_frequency_map, masked_image_embeds, pose_embeds],dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, condition_latent, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(train_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(train_params)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(train_params)

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.validation_prompt is not None:
            edited_images = []
            pipeline = pipeline.to(accelerator.device)
            with torch.autocast(str(accelerator.device).replace(":0", "")):
                for _ in range(args.num_validation_images):
                    edited_images.append(
                        pipeline(
                            args.validation_prompt,
                            image=original_image,
                            num_inference_steps=20,
                            image_guidance_scale=1.5,
                            guidance_scale=7,
                            generator=generator,
                        ).images[0]
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
