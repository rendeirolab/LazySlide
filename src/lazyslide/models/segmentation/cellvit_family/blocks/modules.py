"""CellViT decoder utilities."""

import torch
from torch import nn
from torch.nn import functional as F


class Conv2DBlock(nn.Module):
    """2D conv followed by BN, ReLU activation and dropout.

    Parameters
    ----------
    in_channels : int
        Number of input channels for convolution
    out_channels : int
        Number of output channels for convolution
    kernel_size : int, optional
        Kernel size for convolution. Defaults to 3.
    dropout : float, optional
        Dropout. Defaults to 0.
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
        """Forward pass through the block."""
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d for upsampling.

    The ConTranspose2d is followed by Conv2d, batch-normalisation,
    ReLU activation and dropout

    Parameters
    ----------
    in_channels : int
        Number of input channels for convolution
    out_channels : int
        Number of output channels for convolution
    kernel_size : int, optional
        Kernel size for convolution. Defaults to 3.
    dropout : float, optional
        Dropout. Defaults to 0.
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
        """Forward pass through the block."""
        return self.block(x)


class DecoderBranch(nn.Module):
    """Module for a decoder upsampling branch.

    Parameters.
    ----------
    num_classes : int
        Number of classes for the output maps
    embed_dim : int
        Embedding dimension of the Vision Transformer
    bottleneck_dim : int
        Dimension at the bottleneck
    image_size : int
        Image size that passed to the decoding branch.
    patch_size : int
        Patch size of the ViT encoder.
    dropout : float, optional
        Dropout. Defaults to 0.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        bottleneck_dim: int,
        image_size: int,
        patch_size: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(bottleneck_dim * 2, bottleneck_dim, dropout=dropout),
            Conv2DBlock(bottleneck_dim, bottleneck_dim, dropout=dropout),
            Conv2DBlock(bottleneck_dim, bottleneck_dim, dropout=dropout),
            nn.ConvTranspose2d(
                in_channels=bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=dropout),
            Conv2DBlock(256, 256, dropout=dropout),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=dropout),
            Conv2DBlock(128, 128, dropout=dropout),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=dropout),
            Conv2DBlock(64, 64, dropout=dropout),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        expected_size = get_expected_spatial_dimensions_of_upsampled_volume(
            image_size, patch_size
        )

        self.target_size = (image_size, image_size)
        self.upsampled_size = (expected_size, expected_size)

    def forward(self, z4, n):
        """Apply an upsampling branch to get the output map.

        Uses the bottleneck features (z4) and the spatial features from the neck (n).

        Parameters
        ----------
        z4 : torch.Tensor
            Feature map at the deepest level of the Vision Transformer.
            Input for the bottleneck upsampling path.
            shape [B, emb_dim, H//16, W//16]
        n : list[torch.Tensor]
            Transformed features adapted for the upsampling path.
            n = [n0, n1, n2, n3]
            n0 : [B, 64, H, W]
            n1 : [B, 128, H//2, W//2]
            n2 : [B, 256, H//4, W//4]
            n3 : [B, bottleneck_dim, H//8, W//8]
        branch_decoder (nn.Sequential): Branch decoder network

        Returns
        -------
        branch_output : torch.Tensor
            output log probabilities for each class.
            shape = [B, num_classes, H, W]
        """
        ## Bottle neck upsampling
        b4 = self.bottleneck_upsampler(z4)
        ## Skip co 3 + upsampling
        b3 = n[3]
        b3 = self.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        ## Skip co 2 + upsampling
        b2 = n[2]
        b2 = self.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        ## Skip co 1 + upsampling
        b1 = n[1]
        b1 = self.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        ## Skip co 0 + header projection
        b0 = n[0]
        # In the case of ViT with patch_size 14, the output maps have size (256, 256).
        # They need to be downsampled to the input shape (224, 224).
        b1 = downscale_map(b1, self.upsampled_size, self.target_size)
        b0 = self.decoder0_header(torch.cat([b0, b1], dim=1))

        return b0


class CellViTNeck(nn.Module):
    """CellViTNeck module.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension of the Vision Transformer.
    skip_dim_1 : int
        Number of channels in the first skip connection.
    skip_dim_2 : int
        Number of channels in the second skip connection.
    bottleneck_dim : int
        Number of channels in the bottleneck layer.
    dropout : float, optional
        Dropout. Defaults to 0.
    """

    def __init__(
        self,
        embed_dim: int,
        skip_dim_1: int,
        skip_dim_2: int,
        bottleneck_dim: int,
        dropout: float = 0,
    ) -> None:
        self.embed_dim = embed_dim
        self.skip_dim_1 = skip_dim_1
        self.skip_dim_2 = skip_dim_2
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout

        super().__init__()
        neck0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.dropout),
            Conv2DBlock(32, 64, 3, dropout=self.dropout),
        )
        neck1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_1, dropout=self.dropout),
            Deconv2DBlock(self.skip_dim_1, self.skip_dim_2, dropout=self.dropout),
            Deconv2DBlock(self.skip_dim_2, 128, dropout=self.dropout),
        )
        neck2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_1, dropout=self.dropout),
            Deconv2DBlock(self.skip_dim_1, 256, dropout=self.dropout),
        )
        neck3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.dropout),
        )
        self.necks = nn.ModuleList([neck0, neck1, neck2, neck3])

    def forward(self, z):
        """Forward pass of the CellViTNeck.

        Parameters
        ----------
        z : torch.Tensor
            Output features at 4 depths of the Vision Transformer.
            z = [z0, z1, z2, z3]
            z0 : [B, 3, H, W]
            zi : [B, embed_dim, H//16, W//16] for i=1,2,3

        Returns
        -------
        torch.Tensor
            Transformed features adapted for the upsampling path.
            n = [n0, n1, n2, n3]
            n0 : [B, 64, H, W]
            n1 : [B, 128, H//2, W//2]
            n2 : [B, 256, H//4, W//4] for i=1,2
            n3 : [B, bottleneck_dim, H//8, W//8]
        """
        n = []
        for i in range(4):
            n.append(self.necks[i](z[i]))
        return n


class MLP(nn.Module):
    """MLP layer.

    This module is mainly used for tissue classification.

    Parameters.
    ----------
    in_dim : int
        Input dimension
    hidden_dim : int, optional
        Hidden dimension. Defaults to None.
    out_dim : int, optional
        Output dimension. Defaults to None.
    dropout : float, optional
        Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """Forward pass through the MLP layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor : [B, in_dim]

        Returns
        -------
        torch.Tensor
            Output tensor : [B, out_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


def get_expected_spatial_dimensions_of_upsampled_volume(
    image_size: int, patch_size: int
) -> int:
    """Compute the expected spatial dimension of the upsampled volumne after decoding.

    Parameters
    ----------
    image_size : int
        Spatial dimension (image size) of the input volume in the decoder.

    patch_size : int
        Patch size of the encoder.

    Returns
    -------
    expected_size : int
        Spatial dimension (image size) of the output volume after upscaling.
    """
    assert image_size % patch_size == 0, (
        f"Image size ({image_size}) is not divisible by the patch size ({patch_size})"
    )
    return (image_size // patch_size) * (2**4)


def downscale_map(
    pred: torch.Tensor,
    expected_shape: tuple[int, int] = (256, 256),
    desired_shape: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Downscale output maps to desired shape.

    Useful for ViT with patch_size 14 (Hibou, Virchow, Bioptimus-H0)
    for which the output maps need to be rescaled to the input size.

    Parameters
    ----------
    pred: torch.Tensor
        The prediction map tensor.
    expected_shape: tuple[int] = (256, 256)
        Potential shape of the map.
    desired_shape: tuple[int] = (224, 224)
        Desired shape. Correspond to the input shape.

    Returns
    -------
    torch.Tensor
        The rescaled prediction tensor.
    """
    output_shape = tuple(pred.shape[-2:])
    if output_shape == expected_shape:
        pred = F.interpolate(
            pred, size=desired_shape, mode="bilinear", align_corners=False
        )
    else:
        assert output_shape == desired_shape, (
            f"Shapes of predicted maps are {output_shape}. It should be {desired_shape}!"
        )
    return pred
