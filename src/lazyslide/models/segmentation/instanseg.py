from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torchvision.transforms.v2 import ToImage, ToDtype, Compose

from lazyslide.models.base import SegmentationModel
from .postprocess import instanseg_postprocess


class PercentileNormalize:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image shape should be [C, H, W]
        for c in range(image.shape[0]):
            channel = image[c]
            min_i = torch.quantile(channel.flatten(), 0.001)
            max_i = torch.quantile(channel.flatten(), 0.999)
            image[c] = (channel - min_i) / max(1e-3, max_i - min_i)
        return image

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Instanseg(SegmentationModel):
    """Apply the InstaSeg model to the input image."""

    _base_mpp = 0.5

    def __init__(self, model_file=None):
        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", "instanseg/instanseg_v0_1_0.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def get_transform(self):
        return Compose(
            [
                ToImage(),  # Converts numpy or PIL to torch.Tensor in [C, H, W] format
                ToDtype(dtype=torch.float32, scale=False),
                PercentileNormalize(),
            ]
        )

    def segment(self, image):
        # with torch.inference_mode():
        out = self.model(image)
        return out.squeeze().cpu().numpy().astype(np.uint16)

    def get_postprocess(self) -> Callable | None:
        return instanseg_postprocess
