#!/usr/bin/env bash

# sudo fuser -v /dev/nvidia* | grep -o '[[:digit:]]*' |xargs -I{} sudo kill -9 {}

# buck build  --config client.skip-action-graph-cache=true @mode/opt -c python.native_link_strategy=separate \
buck build  @mode/opt @mode/inplace \
  //vision/fair/mae/... --show-output

# 0: pretrain, 1: finetune, 2: test

if [ "$1" -lt 1 ]
then

  echo "pretrain"

  /data/users/"${USER}"/fbsource/fbcode/buck-out/gen/vision/fair/mae/run_pretrain_bin.par \
  --decoder_attn AttentionOrg --encoder_attn AttentionOrg \
  --decoder_num_heads 16 \
  --batch_size 2 --decoder_embed_dim 512 --decoder_depth 4 \
  --epochs 400 --mask_ratio 0.90 --repeat_aug 2 --video \
  --model mae_vit_large_patch16 \
  --sampling_rate 4 --num_frames 16 \
  --mask_type st \
  --num_workers 2 \
  --bias_wd \
  --mask_schedule "cos" --mask_ratio 0.90 --mask_ratio_end 0.99 \
  --trunc_init \
  --fp32 \
  --knn_monitor --knn_period 1 \
  --t_patch_size 2 \
  --fp32 \
  --mask_type st --mask_schedule const --mask_ratio 0.95 --repeat_aug 4 --sampling_rate 4 --norm_pix_loss \
  --resume manifold://winvision/tree/haoqifan/logs/2022-04-22-212409-469/pretrain/checkpoint-00064.pth \

  --sep_pos_embed \
  --learnable_pos_embed \

  --decoder_attn AttentionRelPos --encoder_attn AttentionRelPos --rel_pos_embed \

else

  if [ "$1" -lt 2 ]
  then

    echo "finetune"

    # AttentionSubsampleMaxpool, AttentionSubsampleStride2, AttentionSubsampleRand10, AttentionSubsampleRand25, AttentionSubsampleRand50,
    /data/users/"${USER}"/fbsource/fbcode/buck-out/gen/vision/fair/mae/run_finetune_bin.par \
    --batch_size 1 --epochs 1 --repeat_aug 1 --video --smoothing 0.1 \
    --mixup 0.0 --cutmix 0.0 --mixup_prob 0.0 \
    --model vit_large_patch16 \
    --t_patch_size 4 --num_frames 16 \
    --rand_aug \
    --encoder_attn AttentionRelPos \
    --rel_pos_init_std 1.0 \
    --sep_pos_embed \
    --fp32 \

    --finetune manifold://winvision/tree/haoqifan/logs/2022-02-05-204420-480/pretrain/checkpoint-00399.pth \

    # --no_qkv_bias

    # --encoder_attn AttentionSubsampleRand10 \

    # --finetune manifold://fair_logging/tree/haoqifan/logs/2022-01-17-162701-592/pretrain/checkpoint-399.pth

  else

    echo "test"

    # AttentionSubsampleMaxpool, AttentionSubsampleStride2, AttentionSubsampleRand10, AttentionSubsampleRand25, AttentionSubsampleRand50,
    /data/users/"${USER}"/fbsource/fbcode/buck-out/gen/vision/fair/mae/run_test_bin.par \
    --batch_size 2 --encoder_attn AttentionSubsampleRand10 \
    --model vit_large_patch16 \
    --t_patch_size 4 --num_frames 16 \
    --finetune manifold://fair_logging/tree/haoqifan/logs/2022-01-25-012936-625/downstream/checkpoint-99.pth

    # --finetune manifold://fair_logging/tree/haoqifan/logs/2022-01-17-162701-592/pretrain/checkpoint-399.pth

  fi
fi
