import mae.util.logging as logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.einsumfunc import _update_other_results
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Attention, DropPath, Mlp
from torch.nn.init import trunc_normal_


logger = logging.get_logger(__name__)


if False:
    import opt_einsum

    def _contract(*args):
        return opt_einsum.contract(*args, backend="torch")


else:
    _contract = torch.einsum


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        # temporal related:
        frames=32,
        t_patch_size=4,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        print(
            f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}"
        )
        self.img_size = img_size
        self.patch_size = patch_size

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        kernel_size = [t_patch_size] + list(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.frames
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        return x


class BlockAttn(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def spatial_attention(q, k, v, heads, scale):
    """
    q: [N, T, S, D]
    k, v: [N, T, R, D], R: spatial size of k and v, can be different from S
    TODO: speed to be optimized
    """
    N, T, S, D = q.shape
    _, _, R, _ = k.shape
    q = q.view([N, T, S, heads, D // heads])
    k = k.view([N, T, R, heads, D // heads])
    v = v.view([N, T, R, heads, D // heads])

    # in this einsum:
    # h: heads
    # q: S of q
    # k: S of k
    attn = torch.einsum("ntqhd,ntkhd->ntqkh", q, k)  # [N, T, S, S, heads]
    attn *= scale
    attn = attn.softmax(dim=-2)  # along the axis of S of k

    # in this einsum:
    # h: heads
    # q: S of q
    # k: S of k (same as S of v)
    x = torch.einsum("ntqkh,ntkhd->ntqhd", attn, v)
    x = x.flatten(-2)  # => [N, T, S, D]
    return x


def temporal_attention(q, k, v, heads, scale):
    """
    q, k, v: each is [N, T, S, D]
    TODO: speed to be optimized
    """
    N, T, S, D = q.shape
    _, _, R, _ = k.shape
    q = q.view([N, T, S, heads, D // heads])
    k = k.view([N, T, R, heads, D // heads])
    v = v.view([N, T, R, heads, D // heads])

    # in this einsum:
    # h: heads
    # q: T of q
    # k: T of k
    attn = torch.einsum("nqshd,nkshd->nqksh", q, k)  # [N, T, T, S, heads]
    attn *= scale
    attn = attn.softmax(dim=-3)  # along the axis of S of k

    # in this einsum:
    # h: heads
    # q: T of q
    # k: T of k (same as T of v)
    x = torch.einsum("nqksh,nkshd->nqshd", attn, v)
    x = x.flatten(-2)  # => [N, T, S, D]
    return x


def spatiotemporal_attention(q, k, v, heads, scale):
    """
    q: [N, T, S, D] or [N, T*S, D]
    k, v: [N, T, R, D] or [N, T*R, D]
    TODO: speed to be optimized
    """
    q_shape = q.shape
    N, D = q.shape[0], q.shape[-1]
    q = q.view([N, -1, heads, D // heads])
    k = k.view([N, -1, heads, D // heads])
    v = v.view([N, -1, heads, D // heads])

    # in this einsum:
    # h: heads
    # q: L of q
    # k: L of k
    attn = torch.einsum("nqhd,nkhd->nqkh", q, k)  # [N, L, L, heads]
    attn *= scale
    attn = attn.softmax(dim=-2)  # along the axis of L of k

    # in this einsum:
    # h: heads
    # q: L of q
    # k: L of k (same as L of v)
    x = torch.einsum("nqkh,nkhd->nqhd", attn, v)
    x = x.reshape(q_shape)  # => [N, T, S, D] or [N, T*S, D]
    # x = x.flatten(-2)  # => [N, T, S, D]
    return x


# ------------------------------------
# MSA block implementation
# ------------------------------------
class AttentionOrg(Attention):
    # for compatibility
    def forward(self, x, **kwargs):
        return super(AttentionOrg, self).forward(x)


class AttentionFactorizeDotProd(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.requires_t_shape = True  # requires temporal shape

    def forward(self, x, kv_subsample=False):
        N, T, S, D = x.shape  # S: H*W
        qkv = self.qkv(x)  # [N, T, S, 3 * D]

        # q, k, v: each is [N, T, S, D]
        q_s, q_t, k_s, k_t, v_s, v_t = qkv.split(D // 2, dim=-1)

        if kv_subsample:
            # spatial subsampling by max pooling for now
            # note: pool3d works for [N, C, T, H, W] order, here we have the [N, T, H, W, C] tensor.
            # we adpat the kernel accordingly, as we only pool H and W
            def subsample_func(y):
                kernel_size = [2, 2, 1]
                dim = y.shape[-1]
                y = torch.nn.functional.max_pool3d(
                    y.reshape([N, T, int(S ** 0.5), -1, dim]),
                    kernel_size=kernel_size,
                    stride=kernel_size,
                ).view([N, T, -1, dim])
                return y

            k_s = subsample_func(k_s)
            v_s = subsample_func(v_s)

        x_s = spatial_attention(
            q_s, k_s, v_s, heads=self.num_heads // 2, scale=self.scale
        )
        x_t = temporal_attention(
            q_t, k_t, v_t, heads=self.num_heads // 2, scale=self.scale
        )

        x = torch.cat([x_s, x_t], dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionSubsample(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.requires_t_shape = True  # requires temporal shape

    @staticmethod
    def subsample_func(y):
        raise NotImplementedError

    def forward(self, x):
        N, T, S, D = x.shape  # S: H*W

        qkv = self.qkv(x)  # [N, T, S, 3 * D]

        q, k, v = qkv.split(D, dim=-1)

        k = self.subsample_func(k)
        v = self.subsample_func(v)

        x = spatiotemporal_attention(q, k, v, heads=self.num_heads, scale=self.scale)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionSubsampleMaxpool(AttentionSubsample):
    # max pool subsample
    @staticmethod
    def subsample_func(y):
        N, T, S, D = y.shape  # S: H*W
        # spatial subsampling by max pooling
        # note: pool3d works for [N, C, T, H, W] order, here we have the [N, T, H, W, C] tensor.
        # we adpat the kernel accordingly, as we only pool H and W
        kernel_size = [2, 2, 1]
        y = torch.nn.functional.max_pool3d(
            y.reshape([N, T, int(S ** 0.5), -1, D]),
            kernel_size=kernel_size,
            stride=kernel_size,
        ).view([N, T, -1, D])
        return y


class AttentionSubsampleStride2(AttentionSubsample):
    # stride=2 subsample, with random 0/1 offsets
    @staticmethod
    def subsample_func(y):
        N, T, S, D = y.shape  # S: H*W
        y = y.reshape([N, T, int(S ** 0.5), -1, D])
        i = torch.randint(
            high=2,
            size=[
                1,
            ],
        )
        j = torch.randint(
            high=2,
            size=[
                1,
            ],
        )
        y = y[:, :, i::2, j::2, :]
        y = y.reshape([N, T, -1, D])
        return y


def rand_subsample_func(y, ratio=0.25):
    N, T, S, D = y.shape  # S: H*W
    len_keep = int(S * ratio)
    with torch.no_grad():
        noise = torch.rand(N, T, S, device=y.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(
            noise, dim=-1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :, :len_keep]

    y = torch.gather(y, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))
    return y


class AttentionSubsampleRand25(AttentionSubsample):
    # random subsample 25%
    @staticmethod
    def subsample_func(y):
        return rand_subsample_func(y, ratio=0.25)


class AttentionSubsampleRand50(AttentionSubsample):
    # random subsample 50%
    @staticmethod
    def subsample_func(y):
        return rand_subsample_func(y, ratio=0.50)


class AttentionSubsampleRand10(AttentionSubsample):
    # random subsample 10%
    @staticmethod
    def subsample_func(y):
        return rand_subsample_func(y, ratio=0.10)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(
        B,
        T // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = torch.einsum("btmhywxc->bthwmyxc", x)
    windows = windows.contiguous().view(
        -1, window_size[0], window_size[1], window_size[2], C
    )
    return windows


def window_reverse(windows, window_size, T, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(
        windows.shape[0]
        / (T * H * W / window_size[0] / window_size[1] / window_size[2])
    )
    x = windows.view(
        B,
        T // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = torch.einsum("bthwmyxc->btmhywxc", x)
    x = x.contiguous().view(B, T, H, W, -1)
    return x


class AttentionSwinInd(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ind=0,
        window_size=(4, 7, 7),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ind = ind
        self.window_size = window_size

        self.requires_t_shape = True  # requires temporal shape

    def forward(self, x):
        N, T, S, D = x.shape
        H = W = int(S ** 0.5)
        x = x.view(N, T, H, W, D)

        window_size, shift_size, reverse_shift_size = self.get_window_and_shift_size(
            self.ind,
            self.window_size,
            (T, H, W),
        )

        shifted_x = torch.roll(x, shifts=shift_size, dims=(1, 2, 3))

        # partition windows
        x_windows = window_partition(
            shifted_x, window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, window_size[0] * window_size[1] * window_size[2], D
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(
            -1, window_size[0], window_size[1], window_size[2], D
        )
        shifted_x = window_reverse(attn_windows, window_size, T, H, W)  # B H' W' C

        x = torch.roll(shifted_x, shifts=reverse_shift_size, dims=(1, 2, 3))
        x = x.view(N, T, S, D)
        return x

    def attn(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def get_window_and_shift_size(ind, window_size, video_size):
        raise NotImplementedError


class AttentionSwinIndNoShift(AttentionSwinInd):
    @staticmethod
    def get_window_and_shift_size(ind, window_size, video_size):
        shift_size = (0, 0, 0)
        reverse_shift_size = (0, 0, 0)
        return window_size, shift_size, reverse_shift_size


class AttentionSwinIndShift(AttentionSwinInd):
    @staticmethod
    def get_window_and_shift_size(ind, window_size, video_size):
        if ind % 2 == 0:
            shift_size = list((i // 2 for i in (window_size)))
            reverse_shift_size = list((-i // 2 for i in (window_size)))
        else:
            shift_size = (0, 0, 0)
            reverse_shift_size = (0, 0, 0)
        return window_size, shift_size, reverse_shift_size


class AttentionSwinIndShift2Global1(AttentionSwinInd):
    @staticmethod
    def get_window_and_shift_size(ind, window_size, video_size):
        if ind % 3 == 0:
            window_size = video_size
            shift_size = (0, 0, 0)
            reverse_shift_size = (0, 0, 0)
        else:
            ind = ind % 3
            if ind % 2 == 0:
                shift_size = list((i // 2 for i in (window_size)))
                reverse_shift_size = list((-i // 2 for i in (window_size)))
            else:
                shift_size = (0, 0, 0)
                reverse_shift_size = (0, 0, 0)
        return window_size, shift_size, reverse_shift_size


class AttentionSwinIndShift4Global1(AttentionSwinInd):
    @staticmethod
    def get_window_and_shift_size(ind, window_size, video_size):
        if ind % 5 == 0:
            window_size = video_size
            shift_size = (0, 0, 0)
            reverse_shift_size = (0, 0, 0)
        else:
            ind = ind % 3
            if ind % 2 == 0:
                shift_size = list((i // 2 for i in (window_size)))
                reverse_shift_size = list((-i // 2 for i in (window_size)))
            else:
                shift_size = (0, 0, 0)
                reverse_shift_size = (0, 0, 0)
        return window_size, shift_size, reverse_shift_size


class AttentionSwinIndShift8Global1(AttentionSwinInd):
    @staticmethod
    def get_window_and_shift_size(ind, window_size, video_size):
        if ind % 9 == 0:
            window_size = video_size
            shift_size = (0, 0, 0)
            reverse_shift_size = (0, 0, 0)
        else:
            ind = ind % 3
            if ind % 2 == 0:
                shift_size = list((i // 2 for i in (window_size)))
                reverse_shift_size = list((-i // 2 for i in (window_size)))
            else:
                shift_size = (0, 0, 0)
                reverse_shift_size = (0, 0, 0)
        return window_size, shift_size, reverse_shift_size


def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


def cal_rel_pos_spatial(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w):
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)

    rel_h = _contract("bythwc,hkc->bythwk", r_q, Rh)
    rel_w = _contract("bythwc,wkc->bythwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h[:, :, :, :, :, None, :, None]
        + rel_w[:, :, :, :, :, None, None, :]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = (
        torch.arange(q_t)[:, None] * q_t_ratio - torch.arange(k_t)[None, :] * k_t_ratio
    )
    dist_t += (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)

    rel = _contract("bythwc,tkc->bythwk", r_q, Rt)

    # attn[:, :, 1:, 1:] += attn_t
    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


class AttentionRelPos(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
        rel_pos_init_std=0.02,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.requires_t_shape = True  # requires temporal shape

        # relative positional embedding

        assert input_size[1] == input_size[2]
        q_size = input_size[1]
        kv_size = input_size[1]
        rel_sp_dim = 2 * max(q_size, kv_size) - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_t = nn.Parameter(
            torch.zeros(2 * input_size[0] - 1, dim // num_heads)
        )

        if rel_pos_init_std > 0.0:
            trunc_normal_(self.rel_pos_h, std=rel_pos_init_std)
            trunc_normal_(self.rel_pos_w, std=rel_pos_init_std)
            trunc_normal_(self.rel_pos_t, std=rel_pos_init_std)

    def forward(self, x):
        B, T, S, C = x.shape
        H = W = int(S ** 0.5)
        N = T * S
        x = x.view(B, N, C)

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = cal_rel_pos_spatial(
            attn,
            q,
            False,
            (T, H, W),
            (T, H, W),
            self.rel_pos_h,
            self.rel_pos_w,
        )

        attn = cal_rel_pos_temporal(
            attn,
            q,
            False,
            (T, H, W),
            (T, H, W),
            self.rel_pos_t,
        )

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, T, -1, C)
        return x
