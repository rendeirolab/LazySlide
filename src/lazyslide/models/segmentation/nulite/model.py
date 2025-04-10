from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn
import timm


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

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


class FastViTEncoder(nn.Module):
    def __init__(self, vit_structure, pretrained=True):
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

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        vit_structure (Literal["SAM-B", "SAM-L", "SAM-H"]): SAM model type
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.

    Raises:
        NotImplementedError: Unknown Fast-ViT backbone structure
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
        drop_rate: int = 0,
    ):
        super().__init__()
        self.vit_structure = vit_structure
        self.embed_dims = self.embed_dims_info.get(vit_structure)
        if self.embed_dims is None:
            raise NotImplementedError("Unknown Fast-ViT backbone structure")
        self.drop_rate = drop_rate
        self.num_nuclei_classes = num_nuclei_classes
        self.regression_loss = False
        self.encoder = FastViTEncoder(vit_structure)
        self.classifier_head = (
            nn.Linear(self.embed_dims[-1], num_tissue_classes)
            if num_tissue_classes > 0
            else nn.Identity()
        )
        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, self.embed_dims[-4], 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64

        offset_branches = 0
        self.branches_output = {
            "nuclei_binary_map": 2 + offset_branches,
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

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of FastViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
        """

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        _, z1, z2, z3, z4 = x, *z

        decoder = self._forward_upsample(z1, z2, z3, z4, self.decoder)

        xt = self.decoder0(x)
        xt = torch.cat([xt, decoder], dim=1)
        out_dict["nuclei_binary_map"] = self.np_head(xt)

        out_dict["hv_map"] = self.hv_head(xt)
        out_dict["nuclei_type_map"] = self.tp_head(xt)

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def _forward_upsample(
        self,
        # z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b5 = branch_decoder.bottleneck_upsampler(z4)
        b4 = branch_decoder.decoder4_upsampler(torch.cat([z3, b5], dim=1))
        b3 = branch_decoder.decoder3_upsampler(torch.cat([z2, b4], dim=1))
        b2 = branch_decoder.decoder2_upsampler(torch.cat([z1, b3], dim=1))
        b1 = branch_decoder.decoder1_upsampler(b2)
        return b1

    def create_upsampling_branch(self) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.Sequential(
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

        decoder4_upsampler = nn.Sequential(
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

        decoder3_upsampler = nn.Sequential(
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

        decoder2_upsampler = nn.Sequential(
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

        decoder1_upsampler = nn.Sequential(
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

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder4_upsampler", decoder4_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                ]
            )
        )

        return decoder
