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
import timm
import torch
import torch.nn as nn
from mae.util import video_vit
from mae.util.logging import master_print as print
from mae.util.pos_embed import get_3d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed, Block


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=4,
        patch_embed=video_vit.PatchEmbed,
        encoder_block=video_vit.BlockAttn,
        decoder_block=video_vit.BlockAttn,
        encoder_attn="AttentionOrg",
        decoder_attn="AttentionOrg",
        mask_type=False,
        no_qkv_bias=False,
        learnable_pos_embed=False,
        sep_pos_embed=False,
        trunc_init=False,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.embed_dim = embed_dim

        encoder_attn_func = video_vit.__dict__[encoder_attn]
        decoder_attn_func = video_vit.__dict__[decoder_attn]

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        self.mask_type = mask_type

        print(f"==== num_patches {num_patches}")

        decoder_attn_func = video_vit.__dict__[decoder_attn]
        encoder_attn_func = video_vit.__dict__[encoder_attn]

        self.learnable_pos_embed = learnable_pos_embed
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
            if learnable_pos_embed:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches, embed_dim), requires_grad=True
                )  # fixed sin-cos embedding
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches, embed_dim), requires_grad=False
                )  # fixed sin-cos embedding

        assert "RelPos" not in encoder_attn, "Not support RelPos for MAE model"
        self.blocks = nn.ModuleList(
            [
                encoder_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    attn_func=encoder_attn_func,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, input_size[1] * input_size[2], decoder_embed_dim
                )
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
        else:
            if learnable_pos_embed:
                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=True
                )  # fixed sin-cos embedding
            else:
                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
                )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                decoder_block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    attn_func=decoder_attn_func,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, t_patch_size * patch_size ** 2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)
        else:
            if self.learnable_pos_embed:
                # torch.nn.init
                torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
            else:
                pos_embed = get_3d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    self.patch_embed.grid_size,
                    self.patch_embed.t_grid_size,
                    cls_token=False,
                )
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

                decoder_pos_embed = get_3d_sincos_pos_embed(
                    self.decoder_pos_embed.shape[-1],
                    self.patch_embed.grid_size,
                    self.patch_embed.t_grid_size,
                    cls_token=False,
                )
                self.decoder_pos_embed.data.copy_(
                    torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
                )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            # torch.nn.init.normal_(self.cls_token, std=.02)
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.patch_embed.t_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p ** 2 * 3))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        # add pos embed w/o cls token
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
        if self.mask_type == "st":
            x = x.reshape(N, T * L, C)
        elif self.mask_type == "t":
            x = x.reshape(N * T, L, C)
        elif self.mask_type == "tube":
            x = torch.einsum("ntlc->nltc", x.reshape([N, T, L, C])).reshape(
                [N, L, T * C]
            )
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        if self.mask_type in ["st", "t"]:
            x = x.view(N, -1, C)
        elif self.mask_type == "tube":
            _, L_new, _ = x.shape
            x = x.reshape([N, L_new, T, C])
            x = torch.einsum("nltc->ntlc", x)
            x = x.reshape([N, T * L_new, C])
            # N 1 L C -> N T L C
            mask = mask.repeat(1, T, 1, 1)
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        if self.mask_type == "st":
            x_ = x_.view([N, T * H * W, C])
        elif self.mask_type == "t":
            x_ = x_.view([N * T, H * W, C])
        elif self.mask_type == "tube":
            x_ = x_.reshape([N, T, H * W, C])
            x_ = torch.einsum("ntlc->nltc", x_)
            x_ = x_.reshape([N, H * W, T * C])
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        if self.mask_type in ["st", "t"]:
            x = x_.view([N, T * H * W, C])
        elif self.mask_type == "tube":
            x = x_.reshape([N, H * W, T, C])
            x = torch.einsum("nltc->ntlc", x)
            x = x.reshape([N, T * H * W, C])
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")

        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        # remove cls token
        x = x[:, :, :]

        return x

    def forward_loss(self, imgs, pred, mask, visualize):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if visualize:
            self.target = target
        if self.norm_pix_loss:
            self.mean = target.mean(dim=-1, keepdim=True)
            self.var = target.var(dim=-1, keepdim=True)
            target = (target - self.mean) / (self.var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, index, mask_ratio=0.75, visualize=False, knn_only=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        if knn_only:
            knn_latent = torch.mean(latent, dim=1, keepdim=False)
            knn_latent = torch.nn.functional.normalize(knn_latent, p=2, dim=1)
            return knn_latent

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, visualize)

        if visualize:
            N, T, H, W, p, u, t, h, w = self.patch_info
            pred = pred

            reconstruct = self.unpatchify(
                pred * mask.reshape(N, t * h * w, 1) + self.target * (1 - mask.reshape(N, t * h * w, 1))
            )
            masked = self.unpatchify(self.target * (1 - mask.reshape(N, t * h * w, 1)))
            comparison = torch.stack(
                [imgs, masked, reconstruct],
                dim=1,
            )
            return loss, pred, mask, comparison
        else:
            return loss, pred, mask, torch.Tensor()


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
