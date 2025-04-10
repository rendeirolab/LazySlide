from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from lazyslide.models.base import SegmentationModel
from .postprocess import instanseg_postprocess


def instanseg_preprocess(image: np.ndarray) -> torch.Tensor:
    channel_index = np.argmin(image.shape)
    if channel_index == 0:
        image = image.transpose(1, 2, 0)
    image = image.astype(np.float32).copy()

    for c in range(3):
        min_i, max_i = np.percentile(image[:, :, c], [0.1, 99.9])
        image[:, :, c] = (image[:, :, c] - min_i) / max(1e-3, max_i - min_i)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return image


class Instanseg(SegmentationModel):
    """Apply the InstaSeg model to the input image."""

    _base_mpp = 0.5

    def __init__(self, model_file=None):
        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", "instanseg/instanseg_v0_0_1.pt"
        )

        self.model = torch.jit.load(model_file)

    def get_transform(self):
        return instanseg_preprocess

    def segment(self, image):
        with torch.inference_mode():
            out = self.model(image)
        return out.squeeze().cpu().numpy().astype(np.uint16)

    def get_postprocess(self) -> Callable | None:
        return instanseg_postprocess
