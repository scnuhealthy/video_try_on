# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import mae.util.logging as logging
import mae.util.video_vit as video_vit
import timm.models.vision_transformer
import torch
import torch.nn as nn
from mae.util.logging import master_print as print


class VisionTransformer(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        num_frames,
        t_patch_size,
        encoder_attn,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=400,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        no_qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        dropout=0.5,
        rel_pos_init_std=0.02,
        sep_pos_embed=False,
        **kwargs,
    ):
        super().__init__()
        print(locals())

        self.sep_pos_embed = sep_pos_embed
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = video_vit.PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        print(f"==== num_patches {num_patches}")

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, input_size[1] * input_size[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim), requires_grad=True
            )  # fixed or not?

        encoder_block = video_vit.BlockAttn
        attn_func = video_vit.__dict__[encoder_attn]
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        if "RelPos" in encoder_attn:
            attn_func = partial(
                attn_func,
                input_size=self.patch_embed.input_size,
                rel_pos_init_std=rel_pos_init_std,
            )
        self.blocks = nn.ModuleList(
            [
                encoder_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    attn_func=partial(attn_func, ind=i)
                    if "Ind" in encoder_attn
                    else attn_func,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # head
        self.fc_norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        torch.nn.init.normal_(self.head.weight, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape  # T: temporal; L: spatial

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
        else:
            pos_embed = self.pos_embed[:, :, :]
        x = x.view([N, T * L, C]) + pos_embed

        # add pos embed w/o cls token
        x = x.view([N, T * L, C]) + pos_embed[:, 0:, :]  # 0: no cls token

        # reshape to [N, T, L, C] or [N, T*L, C]
        requires_t_shape = (
            len(self.blocks) > 0  # support empty decoder
            and hasattr(self.blocks[0].attn, "requires_t_shape")
            and self.blocks[0].attn.requires_t_shape
        )
        if requires_t_shape:
            x = x.view([N, T, L, C])

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if requires_t_shape:
            x = x.view([N, T * L, C])

        # classifier
        x = x.mean(dim=1)  # global pool
        x = self.fc_norm(x)
        x = self.dropout(x)
        x = self.head(x)

        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
