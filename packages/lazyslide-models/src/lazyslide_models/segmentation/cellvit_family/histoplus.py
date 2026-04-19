"""Implementation of CellViT."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any, Callable, Literal

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from lazyslide._utils import find_stack_level

from ..._model_registry import register
from ..._utils import hf_access
from ...base import ModelTask, SegmentationModel
from .blocks import (
    CellViTNeck,
    DecoderBranch,
)
from .postprocess import np_hv_postprocess

BIOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
BIOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

try:
    from xformers.ops import SwiGLU

    _xformers_available = True
except (ImportError, ModuleNotFoundError):
    _xformers_available = False

if _xformers_available:

    class SwiGLUFFNFused(SwiGLU):
        """SwiGLUFFNFused as implemented in DINO v2 original paper with xformers."""

        def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            act_layer: Callable[..., nn.Module] | None = None,
            drop: float = 0.0,
            bias: bool = True,
            **kwargs,
        ) -> None:
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
            super().__init__(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                bias=bias,
            )
else:

    class SwiGLUFFNFused(nn.Module):
        """SwiGLUFFNFused — pure PyTorch version (no xformers dependency)."""

        def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            act_layer: Callable[..., nn.Module] | None = None,
            drop: float = 0.0,
            bias: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()

            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            # Apply the same padding logic as the original xformers SwiGLUFFNFused
            hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

            # Use the same layer names as xformers SwiGLU for compatibility
            self.w12 = nn.Linear(in_features, hidden_features * 2, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
            self.drop = nn.Dropout(drop)
            self.activation = act_layer() if act_layer is not None else F.silu

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_proj = self.w12(x)
            x1, x2 = x_proj.chunk(2, dim=-1)
            # Apply SiLU to the first half, multiply by second half (matches xformers)
            x = self.activation(x1) * x2
            x = self.drop(self.w3(x))
            return x


def interpolate_positional_encoding(
    pos_embedding: torch.Tensor | torch.nn.parameter.Parameter | None,
    embed_dim: int,
    old_dims: tuple[int, int],
    new_dims: tuple[int, int],
    patch_size: int,
    has_cls_token: bool,
    num_reg_token: int = 0,
    interpolate_offset: float = 0.1,
):
    """Interpolate positional encoding.

    Parameters
    ----------
    pos_embedding: torch.Tensor
        Original positional embedding.
    embed_dim: int
        Embedding dimension
    old_dims: tuple[int, int]
        Spatial dimensions of the input volume used during training.
    new_dims: tuple[int, int]
        Spatial dimensions of the input volume used during inference.
    patch_size: int
        Patch size.
    has_cls_token: bool
        Whether the positional encoding includes the [CLS] token.
    num_reg_token: int
        Number of register tokens in the positional encoding.
    interpolate_offset: float
        Offset used for interpolation.

    Returns
    -------
    torch.Tensor
        Interpolated positional encoding (with first position being the [CLS] token if
        it exists).
    """
    assert pos_embedding is not None

    if num_reg_token > 0 and not has_cls_token:
        raise Exception(
            "If register tokens are found in the positional encodings, the [CLS] token "
            "should also be there."
        )

    if old_dims == new_dims:
        return pos_embedding

    def _extract_position_embeddings(pos_embedding: torch.Tensor):
        # Extract CLS token position embedding if present
        class_pos_emb = None
        if has_cls_token:
            class_pos_emb = pos_embedding[:, 0]

        # Extract register token position embeddings if present
        reg_pos_emb = None
        if num_reg_token > 0:
            reg_pos_emb = pos_embedding[:, 1 : 1 + num_reg_token]

        # Extract patch position embeddings
        start_idx = 1 + num_reg_token if has_cls_token else 0
        patch_pos_emb = pos_embedding[:, start_idx:]

        return class_pos_emb, reg_pos_emb, patch_pos_emb

    class_pos_emb, reg_pos_emb, patch_pos_emb = _extract_position_embeddings(
        pos_embedding
    )

    old_w, old_h = old_dims
    new_w, new_h = new_dims

    old_wp, old_hp = old_w // patch_size, old_h // patch_size
    new_wp, new_hp = new_w // patch_size, new_h // patch_size

    # We add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    new_wp_off, new_hp_off = new_wp + interpolate_offset, new_hp + interpolate_offset

    interp_pos_emb = F.interpolate(
        patch_pos_emb.reshape(1, old_wp, old_hp, embed_dim).permute(0, 3, 1, 2),
        scale_factor=(new_wp_off / old_wp, new_hp_off / old_hp),
        mode="bicubic",
        antialias=False,
    )

    assert interp_pos_emb.shape[-2] == new_wp
    assert interp_pos_emb.shape[-1] == new_hp

    interp_pos_emb = interp_pos_emb.permute(0, 2, 3, 1).view(1, -1, embed_dim)

    new_pos_emb = interp_pos_emb

    if has_cls_token:
        assert class_pos_emb is not None  # for typing
        if num_reg_token > 0:
            new_pos_emb = torch.cat(
                (class_pos_emb[None], reg_pos_emb, interp_pos_emb), dim=1
            )
        else:
            new_pos_emb = torch.cat((class_pos_emb[None], interp_pos_emb), dim=1)

    new_pos_emb = new_pos_emb.to(pos_embedding.dtype)

    return new_pos_emb


class FeatureExtractor(nn.Module):
    output_layers = (
        "encoder_layer_3",
        "encoder_layer_5",
        "encoder_layer_7",
        "encoder_layer_11",
    )

    def __init__(self, tile_size: int = 224):
        super().__init__()
        self._patch_size: int = 16
        self.embed_dim = 768
        self.patch_size = 14
        self.has_cls_token = False
        self.tile_size = tile_size

        out_indices = [int(x.split("_")[-1]) for x in self.output_layers]

        init_options = {
            "model_name": "vit_base_patch14_reg4_dinov2.lvd142m",
            "img_size": 224,
            "patch_size": 14,
            "init_values": 1e-5,
            "num_classes": 0,
            "dynamic_img_size": True,  # timm automatically interpolates pos encoding
            "mlp_ratio": 4,
            "mlp_layer": SwiGLUFFNFused,
            "features_only": True,  # A call returns all the features
            "out_indices": out_indices,
        }

        self.feature_extractor = timm.create_model(**init_options)
        new_pos_embedding = interpolate_positional_encoding(
            pos_embedding=self.feature_extractor.model.pos_embed,
            embed_dim=self.embed_dim,
            old_dims=(224, 224),
            new_dims=(self.tile_size, self.tile_size),
            patch_size=self.patch_size,
            has_cls_token=self.has_cls_token,
        )

        assert self.tile_size % self.patch_size == 0
        self.feature_extractor.model.pos_embed = Parameter(new_pos_embedding)
        self.feature_extractor.model.patch_embed.grid_size = (
            self.tile_size // self.patch_size,
            self.tile_size // self.patch_size,
        )

    def __call__(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Return features.

        Parameters
        ----------
        images: torch.Tensor
            input of size (n_tiles, n_channels, dim_x, dim_y, )
            Ideally `dim_x == dim_y == 224`.

        Returns
        -------
        features : torch.Tensor
            tensor of size (n_tiles, dim) where dim=384 or 768
            or 1024 if the model is respectively a ViT-S, a ViT-B or a ViT-L.
        """
        features_list = self.feature_extractor(images)
        return OrderedDict(dict(zip(self.output_layers, features_list, strict=False)))  # type: ignore


class HistoPLUSModel(nn.Module):
    """Implementation of HistoPLUS CellViT segmentor."""

    def __init__(self, inference_image_size, backbone_tile_size: int = 224) -> None:
        super().__init__()

        self.backbone = FeatureExtractor(tile_size=backbone_tile_size)

        self.patch_size = self.backbone.patch_size

        self.embed_dim = 768
        self.skip_dim_1 = 512
        self.skip_dim_2 = 256
        self.bottleneck_dim = 512

        self.neck = CellViTNeck(
            embed_dim=self.embed_dim,
            skip_dim_1=self.skip_dim_1,
            skip_dim_2=self.skip_dim_2,
            bottleneck_dim=self.bottleneck_dim,
        )

        self.np_branch = DecoderBranch(
            num_classes=2,
            embed_dim=self.embed_dim,
            bottleneck_dim=self.bottleneck_dim,
            image_size=inference_image_size,
            patch_size=self.patch_size,
        )

        self.hv_branch = DecoderBranch(
            num_classes=2,
            embed_dim=self.embed_dim,
            bottleneck_dim=self.bottleneck_dim,
            image_size=inference_image_size,
            patch_size=self.patch_size,
        )

        self.tp_branch = DecoderBranch(
            num_classes=15,  # self.number_cell_types
            embed_dim=self.embed_dim,
            bottleneck_dim=self.bottleneck_dim,
            image_size=inference_image_size,
            patch_size=self.patch_size,
        )

    def _extract_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], Any]:
        """Extract features from the backbone and neck."""
        feature_maps = self.backbone(x)
        assert len(feature_maps) == 4
        z = [x, *list(feature_maps.values())]
        n = self.neck(z[:-1])
        return z, n

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Do forward pass for cell detection and classification.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W].

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Dictionary containing the outputs of the different branches.
            - "np" : torch.Tensor : [B, 2, H, W]
            - "hv" : torch.Tensor : [B, 2, H, W]
            - "tp" : torch.Tensor : [B, number_cell_types, H, W]
        """
        z, n = self._extract_features(x)

        out_dict = OrderedDict()
        out_dict["np"] = self.np_branch(z[-1], n)
        out_dict["hv"] = self.hv_branch(z[-1], n)
        out_dict["tp"] = self.tp_branch(z[-1], n)

        return out_dict


@register(
    key="histoplus",
    is_gated=True,
    task=ModelTask.segmentation,
    license="CC-BY-NC-ND-4.0",
    description="Towards Comprehensive Cellular Characterisation of H&E slides",
    commercial=False,
    github_url="https://github.com/owkin/histoplus",
    hf_url="https://huggingface.co/Owkin-Bioptimus/histoplus",
    paper_url="https://doi.org/10.48550/arXiv.2508.09926",
    bib_key="Adjadj2025-hn",
    param_size="47.9M",
    flops="3.81T",
)
class HistoPLUS(SegmentationModel):
    """
    The output classes are:

    - 0: Background
    - 1: Cancer cell
    - 2: Lymphocytes
    - 3: Fibroblasts
    - 4: Plasmocytes
    - 5: Eosinophils
    - 6: Neutrophils
    - 7: Macrophages
    - 8: Muscle Cell
    - 9: Endothelial Cell
    - 10: Red blood cell
    - 11: Epithelial
    - 12: Apoptotic Body
    - 13: Mitotic Figures
    - 14: Minor Stromal Cell

    """

    _backbone_tile_size = {
        "20x": 224,
        "40x": 448,
    }

    def __init__(
        self,
        tile_size: int = 840,
        magnification: Literal["20x", "40x"] = "20x",
        model_path=None,
        token=None,
    ):
        from huggingface_hub import hf_hub_download

        self.variant = magnification
        self.model = HistoPLUSModel(
            inference_image_size=tile_size,
            backbone_tile_size=self._backbone_tile_size[magnification],
        )
        with hf_access("Owkin-Bioptimus/histoplus"):
            weights = hf_hub_download(
                "Owkin-Bioptimus/histoplus",
                f"histoplus_cellvit_segmentor_{magnification}.pt",
            )
        state_dict = torch.load(weights, map_location="cpu")
        state_dict = remap_state_dict(state_dict)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=BIOPTIMUS_MEAN, std=BIOPTIMUS_STD),
            ]
        )

    def segment(self, image):
        with torch.inference_mode():
            output = self.model(image)
        # return output
        # postprocess the output
        flattened = [
            dict(zip(output.keys(), values)) for values in zip(*output.values())
        ]

        instances_maps = []
        prob_maps = []
        for batch in flattened:
            instance_map = np_hv_postprocess(
                batch["np"].softmax(0).detach().cpu().numpy()[1],
                batch["hv"].detach().cpu().numpy(),
                variant=self.variant,
            )  # Numpy array
            prob_map = batch["tp"].softmax(0).detach().cpu().numpy()  # Skip background
            instances_maps.append(instance_map)
            prob_maps.append(prob_map)

        return {
            "instance_map": np.array(instances_maps),
            "class_map": np.array(prob_maps),
        }

    def supported_output(self):
        return ["instance_map", "class_map"]

    @staticmethod
    def get_classes():
        return {
            0: "Background",
            1: "Cancer cell",
            2: "Lymphocytes",
            3: "Fibroblasts",
            4: "Plasmocytes",
            5: "Eosinophils",
            6: "Neutrophils",
            7: "Macrophages",
            8: "Muscle Cell",
            9: "Endothelial Cell",
            10: "Red blood cell",
            11: "Epithelial",
            12: "Apoptotic Body",
            13: "Mitotic Figures",
            14: "Minor Stromal Cell",
        }

    @classmethod
    def check_input_tile(cls, tile_spec) -> bool:
        check_mpp = tile_spec.mpp == 0.5 or tile_spec.mpp == 0.25
        assert tile_spec.height == tile_spec.width, (
            "HistoPLUS model only supports square tiles."
        )
        # Tile size must be divisible by 14
        assert tile_spec.height % 14 == 0, "Tile size must be divisible by 14."
        assert tile_spec.width % 14 == 0, "Tile size must be divisible by 14."
        if not check_mpp:
            warnings.warn(
                f"To optimize the performance of HistoPLUS model, "
                f"the tiles should be created at the mpp=0.5 or 0.25. "
                f"Current tile size is {tile_spec.width}x{tile_spec.height} with {tile_spec.mpp} mpp.",
                stacklevel=find_stack_level(),
            )
        return True


def remap_state_dict(state_dict):
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("backbone.feature_extractor."):
            # Special cases for top level components
            if key in [
                "backbone.feature_extractor.cls_token",
                "backbone.feature_extractor.reg_token",
                "backbone.feature_extractor.pos_embed",
            ]:
                new_key = key.replace(
                    "backbone.feature_extractor.",
                    "backbone.feature_extractor.model.",
                )
            # Handle patch_embed component
            elif key.startswith("backbone.feature_extractor.patch_embed"):
                new_key = key.replace(
                    "backbone.feature_extractor.patch_embed",
                    "backbone.feature_extractor.model.patch_embed",
                )
            # Handle blocks components
            elif key.startswith("backbone.feature_extractor.blocks"):
                new_key = key.replace(
                    "backbone.feature_extractor.blocks",
                    "backbone.feature_extractor.model.blocks",
                )
            # Handle norm component
            elif key.startswith("backbone.feature_extractor.norm"):
                # No need to handle them, since we are only extracting intermediate
                # feature maps. This norm (weights and bias) is applied in the head
                # of the network at the very end.
                continue
            else:
                new_key = key
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
