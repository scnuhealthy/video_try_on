#!/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Need at least 1 parameter to determine number of machines"
  exit
fi

CHECKPOINT_PATH=manifold://winvision/tree/${USER}/logs/$(date -d "${start} + 1 day" +%F-%H%M%S-%3N)
echo "${CHECKPOINT_PATH}"

manifold mkdirs "${CHECKPOINT_PATH#"manifold://"}"
manifold mkdirs "${CHECKPOINT_PATH#"manifold://"}""/pretrain"
manifold mkdirs "${CHECKPOINT_PATH#"manifold://"}""/downstream"

GANG_SCHEDULE=${GANG_SCHEDULE-1}
GANG_AFFINITY=${GANG_AFFINITY-0}
GPU_TYPE=${GPU_TYPE-3}
POSTFIX=${POSTFIX-"benchmark"}
ENT=${ENT-"default_ncg"}
RUN_BENCHMARK=${RUN_BENCHMARK-0}

# Finetune config: AttentionSubsampleMaxpool, AttentionSubsampleStride2, AttentionSubsampleRand10, AttentionSubsampleRand25, AttentionSubsampleRand50, AttentionOrg, AttentionSwinIndShift, AttentionSwinIndNoShift, AttentionSwinIndShift2Global1, AttentionSwinIndShift4Global1, AttentionSwinIndShift8Global1

if [ "$1" -lt 1 ]
then
  FINETUNE_APPENDIX=" --finetune "${4}
else
  FINETUNE_APPENDIX=""
fi
if [ "$2" -lt 1 ]
then
  TESTING_APPENDIX=" --finetune "${4}
else
  TESTING_APPENDIX=""
fi

DOWNSTREAM_ATT=" --encoder_attn AttentionSubsampleMaxpool"
DOWNSTREAM_ATT=" --encoder_attn AttentionRelPos"
# DOWNSTREAM_ATT=" --encoder_attn AttentionSwinIndShift2Global1"
# DOWNSTREAM_ATT=" --encoder_attn AttentionOrg"

# P_CONFIG=${P_CONFIG-"--batch_size 1 --model mae_vit_large_patch16 --fb_env --epochs 100 --mask_ratio 0.95 --distributed --num_frames 16 --decoder_embed_dim 128 --pin_mem --num_workers 14 --decoder_attn_func global --t_patch_size 4 --repeat_aug 32 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 10 --mask_type st"}
# P_CONFIG=${P_CONFIG-"--batch_size 4 --model mae_vit_large_patch16 --fb_env --epochs 100 --distributed --num_frames 16 --decoder_embed_dim 128 --decoder_depth 8 --decoder_num_heads 16 --pin_mem --num_workers 14 --decoder_attn AttentionOrg --t_patch_size 2 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 30 --mask_type st --mask_schedule const --mask_ratio 0.95 --no_qkv_bias --fp32"}
# P_CONFIG=${P_CONFIG-"--batch_size 2 --model mae_vit_large_patch16 --fb_env --epochs 100 --distributed --num_frames 16 --decoder_embed_dim 128 --decoder_depth 8 --decoder_num_heads 16 --pin_mem --num_workers 14 --decoder_attn AttentionOrg --t_patch_size 2 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 30 --mask_type st --mask_schedule const --mask_ratio 0.95 --no_qkv_bias --fp32"}
# P_CONFIG=${P_CONFIG-"--batch_size 2 --model mae_vit_large_patch16 --fb_env --epochs 100 --distributed --num_frames 16 --decoder_embed_dim 512 --decoder_depth 2 --decoder_num_heads 8 --pin_mem --num_workers 14 --decoder_attn AttentionOrg --t_patch_size 2 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 20 --mask_type st --mask_schedule const --mask_ratio 0.95 --fp32 --sep_pos_embed --trunc_init"}
# P_CONFIG=${P_CONFIG-"--batch_size 2 --model mae_vit_large_patch16 --fb_env --epochs 100 --distributed --num_frames 16 --decoder_embed_dim 128 --decoder_depth 8 --decoder_num_heads 16 --pin_mem --num_workers 14 --decoder_attn AttentionOrg --t_patch_size 2 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 20 --mask_type st --mask_schedule const --mask_ratio 0.95 --no_qkv_bias --knn_monitor --knn_period 1 --fp32"}

# P_CONFIG=${P_CONFIG-"--batch_size 2 --model mae_vit_large_patch16 --fb_env --epochs 400 --distributed --num_frames 16 --decoder_embed_dim 256 --decoder_depth 8 --decoder_num_heads 16 --pin_mem --num_workers 14 --decoder_attn AttentionOrg --t_patch_size 2 --repeat_aug 4 --sampling_rate 4 --blr 0e-4 --warmup_epochs 20 --mask_type st --mask_schedule const --mask_ratio 0.9 --no_qkv_bias --fp32 --norm_pix_loss --resume manifold://winvision/tree/haoqifan/logs/2022-03-23-035836-385/pretrain/checkpoint-00399.pth"}
P_CONFIG=${P_CONFIG-"--batch_size 2 --model mae_vit_large_patch16 --fb_env --epochs 400 --distributed --num_frames 16 --decoder_embed_dim 256 --decoder_depth 8 --decoder_num_heads 16 --pin_mem --num_workers 14 --decoder_attn AttentionOrg --t_patch_size 2 --repeat_aug 4 --sampling_rate 4 --blr 0e-4 --warmup_epochs 20 --mask_type st --mask_schedule const --mask_ratio 0.95 --fp32 --norm_pix_loss --resume manifold://winvision/tree/haoqifan/logs/2022-04-22-212409-469/pretrain/checkpoint-00063.pth"}

# P_CONFIG=${P_CONFIG-"--batch_size 4 --model mae_vit_large_patch16 --fb_env --epochs 100 --distributed --num_frames 16 --decoder_embed_dim 128 --pin_mem --num_workers 14 --decoder_attn AttentionOrg --t_patch_size 4 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 20 --mask_type st --mask_schedule const --mask_ratio 0.9 --mask_ratio_end 0.99 --mask_schedule rand"}
# P_CONFIG=${P_CONFIG-"--batch_size 8 --model mae_vit_large_patch16 --fb_env --epochs 800 --mask_ratio 0.95 --distributed --num_frames 16 --decoder_embed_dim 128 --pin_mem --num_workers 14 --decoder_attn_func global --t_patch_size 4 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 80 --mask_type st"}
# P_CONFIG=${P_CONFIG-"--batch_size 1 --model mae_vit_large_patch16 --fb_env --epochs 25 --mask_ratio 0.95 --distributed --num_frames 16 --decoder_embed_dim 128 --pin_mem --num_workers 14 --decoder_attn_func global --t_patch_size 4 --repeat_aug 32 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 3 --mask_type st"}
# P_CONFIG=${P_CONFIG-"--batch_size 8 --model mae_vit_large_patch16 --fb_env --epochs 800 --mask_ratio 0.95 --distributed --num_frames 16 --decoder_embed_dim 128 --pin_mem --num_workers 14 --decoder_attn_func global --t_patch_size 4 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 80 --mask_type st"}
# P_CONFIG=${P_CONFIG-"--batch_size 8 --model mae_vit_large_patch16 --fb_env --epochs 200 --mask_ratio 0.95 --distributed --num_frames 16 --decoder_embed_dim 128 --pin_mem --num_workers 14 --decoder_attn_func global --t_patch_size 4 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss --blr 1.5e-4 --warmup_epochs 20 --mask_type st"}
D_CONFIG=${D_CONFIG-"--rand_aug --epochs 100 --fb_env --repeat_aug 1 --model vit_large_patch16 --batch_size 2 --distributed --dist_eval --video --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --mixup_prob 1.0 --blr 0.015 --num_frames 16 --pin_mem --num_workers 12 --t_patch_size 2 --sampling_rate 4 --dropout 0.0 --warmup_epochs 5 --layer_decay 0.75 --drop_path_rate 0.2 --aa rand-m7-n4-mstd0.5-inc1 --no_qkv_bias --rel_pos_init_std 0.02 --fp32"}${FINETUNE_APPENDIX}${DOWNSTREAM_ATT}
T_CONFIG=${T_CONFIG-"--fb_env --model vit_large_patch16 --batch_size 2 --distributed --dist_eval --video --num_frames 16 --pin_mem --num_workers 12 --t_patch_size 2 --sampling_rate 4 --dropout 0.0 --no_qkv_bias --fp32"}${TESTING_APPENDIX}${DOWNSTREAM_ATT}
# D_CONFIG=${D_CONFIG-"--rand_aug --epochs 75 --fb_env --repeat_aug 1 --model vit_large_patch16 --batch_size 2 --distributed --dist_eval --video --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --mixup_prob 1.0 --blr 0.005 --num_frames 16 --pin_mem --num_workers 12 --t_patch_size 2 --sampling_rate 4 --dropout 0.0 --warmup_epochs 5 --layer_decay 0.75 --drop_path_rate 0.2 --aa rand-m7-n4-mstd0.5-inc1 --rel_pos_init_std 0.02 --fp32 --sep_pos_embed"}${FINETUNE_APPENDIX}${DOWNSTREAM_ATT}
# T_CONFIG=${T_CONFIG-"--fb_env --model vit_large_patch16 --batch_size 2 --distributed --dist_eval --video --num_frames 16 --pin_mem --num_workers 12 --t_patch_size 2 --sampling_rate 4 --dropout 0.0 --fp32 --sep_pos_embed"}${TESTING_APPENDIX}${DOWNSTREAM_ATT}


flow-cli canary mae.mae.workflow@//fblearner/flow/projects/mae:workflow \
--parameters-json '{
    "num_shard_pretrain": '"${1}"',
    "num_shard_finetune": '"${2}"',
    "num_shard_test": '"${3}"',
    "pretrain_config": "'"${P_CONFIG}"'",
    "downstream_config": "'"${D_CONFIG}"'",
    "test_config": "'"${T_CONFIG}"'",
    "output_dir": "'"${CHECKPOINT_PATH}"'",
    "gang_schedule": "'"${GANG_SCHEDULE}"'",
    "gang_affinity": "'"${GANG_AFFINITY}"'",
    "gpu_type": "'"${GPU_TYPE}"'",
    "entitlement": "'"${ENT}"'"}' \
--entitlement "default_ncg" \
--run-as-secure-group "${SECURE_GROUP-vidron}" \
--name "${POSTFIX}||${P_CONFIG}||${1}nodes" \
--mode opt \

# --entitlement "ar_rp_ncg" \
