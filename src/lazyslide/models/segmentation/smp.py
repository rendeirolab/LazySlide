from __future__ import annotations

from typing import Callable

import torch

from lazyslide.models.base import SegmentationModel
from lazyslide.models.segmentation.postprocess import semanticseg_postprocess


class SMPBase(SegmentationModel, abstract=True):
    """This is a base class for any models from segmentation models pytorch"""

    def __init__(
        self,
        arch: str = "unetplusplus",
        encoder_name: str = "timm-efficientnet-b0",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 3,
        **kwargs,
    ):
        try:
            import segmentation_models_pytorch as smp
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install segmentation_models_pytorch to use this model."
            )

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs,
        )

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        # default_fn = smp.encoders.get_preprocessing_fn(
        #     self.encoder_name, self.encoder_weights
        # )

        return Compose(
            [
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # default_fn
            ]
        )

    def get_postprocess(self) -> Callable:
        return semanticseg_postprocess
