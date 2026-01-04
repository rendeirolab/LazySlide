from contextlib import nullcontext
from typing import List

import torch
from PIL import Image
from wsidata import WSIData

from lazyslide import _api
from lazyslide.models import MODEL_REGISTRY
from lazyslide.models.base import ImageGenerationModel


def generate_image(
    wsi: WSIData,
    model: str = "cytosyn",
    prompt_tiles: slice = None,
    tile_key: str = "tiles",
    device: str = None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    num_images_per_tiles: int = 2,
    seed: int = 0,
    **kwargs,
) -> List[Image.Image]:
    device = _api.default_value("device", device)
    amp = _api.default_value("amp", amp)
    autocast_dtype = _api.default_value("autocast_dtype", autocast_dtype)

    generation_model: ImageGenerationModel = MODEL_REGISTRY[model]()
    try:
        generation_model.to(device)
    except:  # noqa: E722
        pass
    # Check if H0-mini features exist
    try:
        feature_key = wsi._check_feature_key("h0-mini", tile_key)
    except KeyError:
        raise KeyError(
            "H0-mini features are needed for image generation with cytosyn model."
        )
    if isinstance(device, torch.device):
        device = device.type
    amp_ctx = torch.autocast(device, autocast_dtype) if amp else nullcontext()
    with amp_ctx, torch.inference_mode():
        opts = dict(
            num_images_per_prompt=num_images_per_tiles,
            seed=seed,
        )
        opts.update(kwargs)
        # Unconditional generation
        if prompt_tiles is None:
            return generation_model.generate(**opts)
        # Conditional generation
        else:
            cls_tokens = wsi[feature_key][prompt_tiles].X[:, :768]
            cls_tokens = torch.tensor(cls_tokens, dtype=torch.float32)
            return generation_model.generate_conditionally(cls_tokens, **opts)
