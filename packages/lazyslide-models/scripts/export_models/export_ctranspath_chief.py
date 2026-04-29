#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gdown==6.0.0",
#   "numpy==2.3.0",
#   "torch>=2.7.1",
#   "torchvision==0.26.0",
# ]
# ///

# %%
from pathlib import Path

from export_utils import export_model, verify_exported

workdir = Path(__file__).parent
checkpoint_dir = workdir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

export_artifacts = workdir / "export_artifacts"
export_artifacts.mkdir(parents=True, exist_ok=True)

CTRANSPATH_EXPORT_PATH = export_artifacts / "CTransPath_exported.pt2"
CHIEF_PATCH_ENCODER_EXPORT_PATH = export_artifacts / "CHIEF_patch_encoder_exported.pt2"
CHIEF_SLIDE_ENCODER_EXPORT_PATH = export_artifacts / "CHIEF_slide_encoder_exported.pt2"

# %%
# Download from Google Drive
import gdown

CTRANSPATH_CKP_URL = (
    "https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view?usp=sharing"
)
CTRANSPATH_CKP_OUT = Path(checkpoint_dir) / "CTransPath_weights.pth"

CHIEF_CKP_URL = (
    "https://drive.google.com/file/d/1_vgRF1QXa8sPCOpJ1S9BihwZhXQMOVJc/view?usp=sharing"
)
CHIEF_CKP_OUT = Path(checkpoint_dir) / "CHIEF_weights.pth"

CHIEF_SLIDE_ENCODER_CKP_URL = (
    "https://drive.google.com/file/d/10bJq_ayX97_1w95omN8_mESrYAGIBAPb/view?usp=sharing"
)
CHIEF_SLIDE_ENCODER_OUT = Path(checkpoint_dir) / "CHIEF_SlideEncoder_weights.pth"

tasks = [
    (CTRANSPATH_CKP_URL, CTRANSPATH_CKP_OUT),
    (CHIEF_CKP_URL, CHIEF_CKP_OUT),
    (CHIEF_SLIDE_ENCODER_CKP_URL, CHIEF_SLIDE_ENCODER_OUT),
]

for URL, OUT in tasks:
    if not OUT.exists():
        print(f"Downloading from {URL} to {OUT}...")
        gdown.download(url=URL, output=str(OUT.resolve()), quiet=False)


# %%
# ---------------------------------------------------------------------------
# Inlined swin_tiny_patch4_window7_224 — extracted from timm-0.5.4 and
# stripped down to only what CTransPath / CHIEF need.
# Key fix vs timm-0.5.4: window_reverse uses integer floor-division (//)
# instead of int() so the batch dimension stays symbolic during torch.export.
# ---------------------------------------------------------------------------
import math
from collections.abc import Iterable
from itertools import repeat
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out

# --- helpers ----------------------------------------------------------------


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    import math as _math

    def norm_cdf(x):
        return (1.0 + _math.erf(x / _math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)  # noqa: E741
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * _math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def lecun_normal_(tensor):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(1.0 / fan_in) / 0.87962566103423978
    trunc_normal_(tensor, std=std)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# --- weight init (matches timm-0.5.4 _init_vit_weights) --------------------


def _init_vit_weights(
    module: nn.Module, name: str = "", head_bias: float = 0.0, jax_impl: bool = False
):
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(
                        module.bias, std=1e-6
                    ) if "mlp" in name else nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# --- Swin Transformer -------------------------------------------------------


def window_partition(x, window_size: int):
    """x: (B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """windows: (num_windows*B, window_size, window_size, C) -> (B, H, W, C)
    Uses // instead of int() so the batch dim stays symbolic during torch.export.
    """
    num_windows = (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            if downsample is not None
            else None
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        embed_layer=None,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if embed_layer is None:
            embed_layer = _PatchEmbed

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        layers = []
        for i_layer in range(self.num_layers):
            layers.append(
                BasicLayer(
                    dim=int(embed_dim * 2**i_layer),
                    input_resolution=(
                        self.patch_grid[0] // (2**i_layer),
                        self.patch_grid[1] // (2**i_layer),
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging
                    if (i_layer < self.num_layers - 1)
                    else None,
                    use_checkpoint=use_checkpoint,
                )
            )
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class _PatchEmbed(nn.Module):
    """Default patch embedding (conv-based), used when embed_layer is None."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# %%
# ---------------------------------------------------------------------------
# CTransPath ConvStem (custom patch embedding from CTransPath authors)
# ---------------------------------------------------------------------------
class ConvStem(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# %%
# ---------------------------------------------------------------------------
# CHIEF slide encoder
# ---------------------------------------------------------------------------
import warnings

import torch.nn as nn

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
    category=UserWarning,
)


class Att_Head(nn.Module):
    def __init__(self, FEATURE_DIM, ATT_IM_DIM):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, ATT_IM_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ATT_IM_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class CHIEF(nn.Module):
    def __init__(
        self,
        gate=True,
        size_arg="large",
        dropout=True,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        **kwargs,
    ):
        super().__init__()
        self.size_dict = {
            "xs": [384, 256, 256],
            "small": [768, 512, 256],
            "big": [1024, 512, 384],
            "large": [2048, 1024, 512],
        }
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(size[1], 2) for _ in range(n_classes)]
        )
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        initialize_weights(self)

        self.att_head = Att_Head(size[1], size[2])
        self.text_to_vision = nn.Sequential(
            nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25)
        )

    def forward(self, x):
        h_ori = x
        A, x = self.attention_net(x)
        A = torch.transpose(A, 1, 0)
        A = torch.nn.functional.softmax(A, dim=1)
        slide_embeddings = torch.mm(A, h_ori)
        return slide_embeddings


# %%
# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def ctranspath():
    """swin_tiny_patch4_window7_224 with CTransPath ConvStem."""
    return SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        embed_layer=ConvStem,
    )


# %%


# %%
# ---------------------------------------------------------------------------
# CTransPath export
# ---------------------------------------------------------------------------
model = ctranspath()
model.head = torch.nn.Identity()
td = torch.load(CTRANSPATH_CKP_OUT, map_location="cpu")
model.load_state_dict(td["model"], strict=True)

image_dynamic_shapes = [{0: torch.export.Dim.AUTO}]

ctranspath_example_input = torch.randn(2, 3, 224, 224)
export_model(
    model,
    ctranspath_example_input,
    CTRANSPATH_EXPORT_PATH,
    dynamic_shapes=image_dynamic_shapes,
)


# %%
# ---------------------------------------------------------------------------
# CHIEF patch encoder export
# ---------------------------------------------------------------------------
model = ctranspath()
model.head = torch.nn.Identity()
td = torch.load(CHIEF_CKP_OUT, map_location="cpu")
model.load_state_dict(td["model"], strict=True)

chief_patch_encoder_example_input = torch.randn(2, 3, 224, 224)
export_model(
    model,
    chief_patch_encoder_example_input,
    CHIEF_PATCH_ENCODER_EXPORT_PATH,
    dynamic_shapes=image_dynamic_shapes,
)


# %%
# ---------------------------------------------------------------------------
# CHIEF slide encoder export
# ---------------------------------------------------------------------------
model = CHIEF(size_arg="small", dropout=True, n_classes=2)
td = torch.load(CHIEF_SLIDE_ENCODER_OUT, map_location="cpu")
if "organ_embedding" in td:
    del td["organ_embedding"]
model.load_state_dict(td, strict=True)

# N patches per slide is variable; feature dim (768) is fixed.
slide_dynamic_shapes = [{0: torch.export.Dim.AUTO}]

chief_slide_encoder_example_input = torch.randn(256, 768)
export_model(
    model,
    chief_slide_encoder_example_input,
    CHIEF_SLIDE_ENCODER_EXPORT_PATH,
    dynamic_shapes=slide_dynamic_shapes,
)


# %%
# ---------------------------------------------------------------------------
# Verification — reload each artifact and compare against original model
# ---------------------------------------------------------------------------
torch.manual_seed(42)
fixed_image = torch.randn(4, 3, 224, 224)
fixed_patch_features = torch.randn(256, 768)

# CTransPath
ctranspath_model = ctranspath()
ctranspath_model.head = torch.nn.Identity()
td = torch.load(CTRANSPATH_CKP_OUT, map_location="cpu")
ctranspath_model.load_state_dict(td["model"], strict=True)
verify_exported(ctranspath_model, CTRANSPATH_EXPORT_PATH, fixed_image, "CTransPath")

# CHIEF patch encoder
chief_patch_model = ctranspath()
chief_patch_model.head = torch.nn.Identity()
td = torch.load(CHIEF_CKP_OUT, map_location="cpu")
chief_patch_model.load_state_dict(td["model"], strict=True)
verify_exported(
    chief_patch_model,
    CHIEF_PATCH_ENCODER_EXPORT_PATH,
    fixed_image,
    "CHIEF patch encoder",
)

# CHIEF slide encoder
chief_slide_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
td = torch.load(CHIEF_SLIDE_ENCODER_OUT, map_location="cpu")
if "organ_embedding" in td:
    del td["organ_embedding"]
chief_slide_model.load_state_dict(td, strict=True)
verify_exported(
    chief_slide_model,
    CHIEF_SLIDE_ENCODER_EXPORT_PATH,
    fixed_patch_features,
    "CHIEF slide encoder",
)
