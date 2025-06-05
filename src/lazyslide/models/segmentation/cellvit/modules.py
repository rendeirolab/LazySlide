import torch
import torch.nn as nn


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


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
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


# def build_model(
#     weights_path=None,
#     img_size=224,
#     arch="vit_base",
#     patch_size=14,
#     layerscale=1e-5,
#     ffn_layer="swiglufused",
#     block_chunks=0,
#     qkv_bias=True,
#     proj_bias=True,
#     ffn_bias=True,
#     dropout_rate=0.0,
#     attention_dropout_rate=0.0,
#     num_register_tokens=4,
#     interpolate_offset=0,
#     interpolate_antialias=True,
# ):
#     vit_kwargs = dict(
#         img_size=img_size,
#         patch_size=patch_size,
#         init_values=layerscale,
#         ffn_layer=ffn_layer,
#         block_chunks=block_chunks,
#         qkv_bias=qkv_bias,
#         proj_bias=proj_bias,
#         ffn_bias=ffn_bias,
#         num_register_tokens=num_register_tokens,
#         interpolate_offset=interpolate_offset,
#         interpolate_antialias=interpolate_antialias,
#         dropout_rate=dropout_rate,
#         attention_dropout_rate=attention_dropout_rate,
#     )
#     model = vision_transformer.__dict__[arch](**vit_kwargs)
#     if weights_path is not None:
#         print(model.load_state_dict(torch.load(weights_path), strict=False))
#     return model
#
#
# class HibouEncoder(nn.Module):
#     def __init__(
#         self,
#         path=None,
#         extract_layers=[6, 12, 18, 24],
#         num_classes=0,
#         dropout_rate=0,
#         attention_dropout_rate=0,
#     ):
#         super().__init__()
#         self.path = path
#         self.encoder = build_model(
#             weights_path=path,
#             img_size=224,
#             arch="vit_large",
#             patch_size=14,
#             layerscale=1e-5,
#             ffn_layer="swiglufused",
#             block_chunks=0,
#             qkv_bias=True,
#             proj_bias=True,
#             ffn_bias=True,
#             num_register_tokens=4,
#             interpolate_offset=0,
#             interpolate_antialias=True,
#             dropout_rate=dropout_rate,
#             attention_dropout_rate=attention_dropout_rate,
#         )
#         self.extract_layers = extract_layers
#         self.head = nn.Linear(1024, num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward(self, x):
#         x = torchvision.transforms.functional.resize(x, (224, 224))
#         output = self.encoder.forward_features(x, return_intermediate=True)
#         intermediate = output["intermediate"]
#         intermediate = [intermediate[i - 1] for i in self.extract_layers]
#         intermediate = [
#             torch.cat((intermediate[i][:, :1], intermediate[i][:, 5:]), dim=1)
#             for i in range(len(intermediate))
#         ]
#         return self.head(output["x_norm_clstoken"]), None, intermediate
