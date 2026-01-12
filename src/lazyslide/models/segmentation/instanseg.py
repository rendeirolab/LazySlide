from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from lazyslide.models.base import ModelTask, SegmentationModel

from .._model_registry import register

if TYPE_CHECKING:
    from wsidata import TileSpec


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


@register(
    key="instanseg",
    task=ModelTask.segmentation,
    license="Apache 2.0",
    description="An embedding-based instance segmentation algorithm optimized for accurate, "
    "efficient and portable cell segmentation.",
    commercial=True,
    github_url="https://github.com/instanseg/instanseg",
    paper_url="https://doi.org/10.48550/arXiv.2408.15954",
    bib_key="Goldsborough2024-oc",
    param_size="3.8M",
    flops="27.55G",
)
class Instanseg(
    SegmentationModel,
):
    """Apply the InstaSeg model to the input image."""

    _base_mpp = 0.5

    def __init__(self, model_file=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", "instanseg/instanseg_v0_1_0.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, ToDtype, ToImage

        return Compose(
            [
                ToImage(),  # Converts numpy or PIL to torch.Tensor in [C, H, W] format
                ToDtype(dtype=torch.float32, scale=False),
                PercentileNormalize(),
            ]
        )

    @torch.inference_mode()
    def segment(self, image):
        # with torch.inference_mode():
        out = self.model(image)
        # Output is a tensor of B, C, H, W
        # But C is always 1, so we can squeeze it
        return {"instance_map": out.long().squeeze(1)}

    def supported_output(self):
        return ("instance_map",)

    def check_input_tile(self, tile_spec: "TileSpec") -> bool:
        check_mpp = tile_spec.mpp == 0.5
        check_size = tile_spec.height == 512 and tile_spec.width == 512
        if not check_mpp or not check_size:
            warnings.warn(
                f"To optimize the performance of Instanseg model, "
                f"the tile size should be 512x512 and the mpp should be 0.5. "
                f"Current tile size is {tile_spec.width}x{tile_spec.height} with {tile_spec.mpp} mpp."
            )
        return True
