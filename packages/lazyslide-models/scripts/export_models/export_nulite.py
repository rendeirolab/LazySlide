#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub>=0.21",
#   "timm>=1.0",
#   "pooch",
#   "torch>=2.7.1",
# ]
# ///

from pathlib import Path

from export_utils import export_model, verify_exported_dict

workdir = Path(__file__).parent
checkpoint_dir = workdir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

export_artifacts = workdir / "export_artifacts"
export_artifacts.mkdir(parents=True, exist_ok=True)

NULITE_T_EXPORT_PATH = export_artifacts / "NuLite_T_exported.pt2"
NULITE_M_EXPORT_PATH = export_artifacts / "NuLite_M_exported.pt2"
NULITE_H_EXPORT_PATH = export_artifacts / "NuLite_H_exported.pt2"

# %%
import pooch

# %%
import torch
import torch.nn as nn

# Download NuLite weights from Hugging Face
nulite_weights = {
    "T": pooch.retrieve(
        "https://zenodo.org/records/13272655/files/NuLite-T-Weights.pth?download=1",
        path=str(checkpoint_dir),
        known_hash=None,
        fname="NuLite-T-Weights.pth",
    ),
    "M": pooch.retrieve(
        "https://zenodo.org/records/13272705/files/NuLite-M-Weights.pth?download=1",
        path=str(checkpoint_dir),
        known_hash=None,
        fname="NuLite-M-Weights.pth",
    ),
    "H": pooch.retrieve(
        "https://zenodo.org/records/13272667/files/NuLite-H-Weights.pth?download=1",
        path=str(checkpoint_dir),
        known_hash=None,
        fname="NuLite-H-Weights.pth",
    ),
}

# %%
from typing import Dict, List, Literal


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class FastViTEncoder(nn.Module):
    def __init__(self, vit_structure, pretrained=True):
        import timm

        super(FastViTEncoder, self).__init__()

        self.fast_vit = timm.create_model(
            f"{vit_structure}.apple_in1k", features_only=True, pretrained=pretrained
        )

        self.avg_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def forward(self, x):
        extracted_layers = self.fast_vit(x)
        return (
            self.avg_pooling(extracted_layers[-1]),
            extracted_layers[-1],
            extracted_layers,
        )


class NuLite(nn.Module):
    """NuFastViT

    Skip connections are shared between branches, but each network has a distinct encoder.
    """

    embed_dims_info = {
        "fastvit_t8": [48, 96, 192, 384],
        "fastvit_t12": [64, 128, 256, 512],
        "fastvit_s12": [64, 128, 256, 512],
        "fastvit_sa12": [64, 128, 256, 512],
        "fastvit_sa24": [64, 128, 256, 512],
        "fastvit_sa36": [64, 128, 256, 512],
        "fastvit_ma36": [76, 152, 304, 608],
    }

    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal[
            "fastvit_t8",
            "fastvit_t12",
            "fastvit_s12",
            "fastvit_sa24",
            "fastvit_sa36",
            "fastvit_sa36",
        ],
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.vit_structure = vit_structure
        embed_dims = self.embed_dims_info.get(vit_structure)
        if embed_dims is None:
            raise NotImplementedError("Unknown Fast-ViT backbone structure")
        self.embed_dims: List[int] = embed_dims
        self.drop_rate = drop_rate
        self.num_nuclei_classes = num_nuclei_classes
        self.regression_loss = False
        self.encoder = FastViTEncoder(vit_structure)
        self.classifier_head = (
            nn.Linear(self.embed_dims[-1], num_tissue_classes)
            if num_tissue_classes > 0
            else nn.Identity()
        )
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, self.embed_dims[-4], 3, dropout=self.drop_rate),
        )

        self.branches_output = {
            "nuclei_binary_map": 2,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }

        self.decoder = self.create_upsampling_branch()
        self.np_head = nn.Sequential(
            Conv2DBlock(
                2 * self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate
            ),
            nn.Conv2d(
                in_channels=self.embed_dims[-4],
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.hv_head = nn.Sequential(
            Conv2DBlock(
                2 * self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate
            ),
            nn.Conv2d(
                in_channels=self.embed_dims[-4],
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.tp_head = nn.Sequential(
            Conv2DBlock(
                2 * self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate
            ),
            nn.Conv2d(
                in_channels=self.embed_dims[-4],
                out_channels=self.num_nuclei_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Images in BCHW style

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type predictions. Shape: (B, num_nuclei_classes, H, W)
        """
        out_dict: Dict[str, torch.Tensor] = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z1, z2, z3, z4 = z[0], z[1], z[2], z[3]

        decoder = self._forward_upsample(z1, z2, z3, z4)

        xt = self.decoder0(x)
        xt = torch.cat([xt, decoder], dim=1)
        out_dict["nuclei_binary_map"] = self.np_head(xt)
        out_dict["hv_map"] = self.hv_head(xt)
        out_dict["nuclei_type_map"] = self.tp_head(xt)

        return out_dict

    def _forward_upsample(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
    ) -> torch.Tensor:
        b5 = self.bottleneck_upsampler(z4)
        b4 = self.decoder4_upsampler(torch.cat([z3, b5], dim=1))
        b3 = self.decoder3_upsampler(torch.cat([z2, b4], dim=1))
        b2 = self.decoder2_upsampler(torch.cat([z1, b3], dim=1))
        b1 = self.decoder1_upsampler(b2)
        return b1

    def create_upsampling_branch(self) -> None:
        self.bottleneck_upsampler = nn.Sequential(
            Conv2DBlock(
                self.embed_dims[-1], self.embed_dims[-2], dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-2],
                out_channels=self.embed_dims[-2],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder4_upsampler = nn.Sequential(
            Conv2DBlock(
                2 * self.embed_dims[-2], self.embed_dims[-2], dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.embed_dims[-2], self.embed_dims[-3], dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-3],
                out_channels=self.embed_dims[-3],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                2 * self.embed_dims[-3], self.embed_dims[-3], dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.embed_dims[-3], self.embed_dims[-4], dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-4],
                out_channels=self.embed_dims[-4],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder2_upsampler = nn.Sequential(
            Conv2DBlock(
                2 * self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-4],
                out_channels=self.embed_dims[-4],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder1_upsampler = nn.Sequential(
            Conv2DBlock(
                self.embed_dims[-4], self.embed_dims[-4], dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.embed_dims[-4],
                out_channels=self.embed_dims[-4],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )


# %%
# Dynamic batch + spatial dims (H, W); FastViT + conv decoder is fully convolutional
dynamic_shapes = [
    {0: torch.export.Dim.AUTO, 2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO}
]

model_variants = [
    ("T", nulite_weights["T"], NULITE_T_EXPORT_PATH),
    ("M", nulite_weights["M"], NULITE_M_EXPORT_PATH),
    ("H", nulite_weights["H"], NULITE_H_EXPORT_PATH),
]

built_models = {}
for variant, weights_path, export_path in model_variants:
    print(f"Building NuLite-{variant}...")

    weights = torch.load(weights_path, map_location="cpu")
    config = weights["config"]

    model = NuLite(
        config["data.num_nuclei_classes"],
        config["data.num_tissue_classes"],
        config["model.backbone"],
    )

    # Remap state dict keys
    state_dict = weights["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("decoder."):
            new_key = key[8:]  # strip "decoder." prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    built_models[variant] = model

    nulite_example_input = torch.randn(2, 3, 256, 256)
    export_model(
        model, nulite_example_input, export_path, dynamic_shapes=dynamic_shapes
    )
    print(f"Exported NuLite-{variant} to {export_path}")


# %%
# --- Verification ---
torch.manual_seed(42)
fixed_input = torch.randn(4, 3, 256, 256)

for variant, _, export_path in model_variants:
    verify_exported_dict(
        built_models[variant],
        export_path,
        fixed_input,
        f"NuLite-{variant}",
    )
