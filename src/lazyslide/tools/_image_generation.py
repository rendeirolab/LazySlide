from contextlib import nullcontext
from typing import List

import torch
from PIL import Image
from wsidata import WSIData

from lazyslide import _api
from lazyslide.models import MODEL_REGISTRY
from lazyslide.models.base import ImageGenerationModel


def image_generation(
    wsi: WSIData = None,
    model: str | ImageGenerationModel = "cytosyn",
    prompt_tiles: slice = None,
    tile_key: str = "tiles",
    device: str = None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    num_images_per_tiles: int = 2,
    seed: int = 0,
    **kwargs,
) -> List[Image.Image]:
    """
    Generation of tile images unconditionally or conditionally.

    Currently only supports cytosyn model, conditionally generation relied on H0-mini features.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : str, default: "cytosyn"
        The image generation model.
    prompt_tiles : slice, default: None
        The tiles to generate images for, please use index to select tiles.
        If None, unconditional generation is performed.
    tile_key : str, default: "tiles"
        Which tile table to use.
    device : str, optional
        The device to use for inference. If not provided, the device will be automatically selected.
    amp : bool, default: False
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, default: torch.float16
        The dtype for automatic mixed precision.
    num_images_per_tiles : int, default: 2
        The number of images to generate for each tile if conditional generation is used.
        Otherwise, it's the total number of images to generate if unconditional generation is used.
    seed : int, default: 0
        The random seed to ensure reproducible image generation (May not work for all models).
    kwargs : dict, optional
        Please refer to the documentation of the specific model for additional parameters.

    Returns
    -------
    :class:`PIL.Image.Image`
        The function returns a list of generated images in PIL format.

    Examples
    --------

    >>> import lazyslide as zs
    >>> # Unconditional generation
    >>> imgs = zs.tl.image_generation()
    >>> # Conditional generation
    >>> wsi = zs.datasets.sample()
    >>> zs.tl.feature_extraction(wsi, "h0-mini")
    >>> imgs = zs.tl.image_generation(wsi, prompt_tiles=slice(0, 2)) # Generate images for the first two tiles

    """
    device = _api.default_value("device", device)
    amp = _api.default_value("amp", amp)
    autocast_dtype = _api.default_value("autocast_dtype", autocast_dtype)

    if isinstance(model, ImageGenerationModel):
        raise NotImplementedError("Currently only supports cytosyn model.")

    generation_model: ImageGenerationModel = MODEL_REGISTRY[model]()
    try:
        generation_model.to(device)
    except:  # noqa: E722
        pass
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
            # Check if H0-mini features exist
            try:
                feature_key = wsi._check_feature_key("h0-mini", tile_key)
            except KeyError:
                raise KeyError(
                    "H0-mini features are needed for image generation with cytosyn model."
                )
            cls_tokens = wsi[feature_key][prompt_tiles].X[:, :768]
            cls_tokens = torch.tensor(cls_tokens, dtype=torch.float32)
            return generation_model.generate_conditionally(cls_tokens, **opts)
