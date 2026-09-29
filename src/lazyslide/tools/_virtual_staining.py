from __future__ import annotations

import re
import tempfile
from typing import TYPE_CHECKING

import cv2
import numpy as np
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale
from wsidata import WSIData

from lazyslide import _api
from lazyslide._const import Key
from lazyslide._utils import default_pbar

if TYPE_CHECKING:
    import torch
    from lazyslide_models import (
        DensePredictionModelProtocol,
        TilePredictionModelProtocol,
    )


def virtual_stain(
    wsi: WSIData,
    model: str | TilePredictionModelProtocol | DensePredictionModelProtocol = "rosie",
    image_key: str | None = None,
    tile_key: str = Key.tiles,
    device: str | None = None,
    amp: bool | None = None,
    autocast_dtype: torch.dtype = None,
    compile: bool | None = None,
    compile_kws: dict | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    pbar: bool = True,
):
    """
    Translate the :term:`H&E` images to :term:`multiplexed images`.

    A new :term:`multi-channel image` will be created and stored in the :term:`WSIData` object.
    The marker name is recorded in the image channel names.

    What gets produced depends on the kind of model, not on its name:

    - A tile prediction model predicts one value per column for each tile. The
      result is a coarse image with one pixel per tile, holding the predicted
      values as float32. ROSIE is the exception: following the ROSIE codebase,
      its output is contrast-stretched per channel into uint8.
    - A marker map model, such as GigaTIME, predicts a value for every pixel.
      Tiles are stitched into one image, blended where they overlap.
    - A virtual stain model predicts one or more RGB stains for every pixel.
      Each stain is stored as its own RGB image.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image data to work on.
    model : str or model, default: "rosie"
        The virtual staining model to use: a registry key, or an instance of a
        tile prediction, marker map or virtual stain model.
    image_key : str, default: None
        The key to store the new image. For a virtual stain model this is a
        prefix, and each stain is stored under ``'{image_key}_{stain}'``.
    tile_key : str, default: "tiles"
        The key for the tile table.
    device : str, optional
        Which device to use for inference. If None, the default device is used.
    amp : bool, optional
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, optional
        The dtype for automatic mixed precision.
    compile : bool, optional
        Whether to compile the model with :func:`torch.compile`.
        Compilation is best-effort and is silently skipped for models
        that do not support it.
    compile_kws : dict, optional
        Keyword arguments passed to :func:`torch.compile`.
    batch_size : int, default: 32
        The batch size for inference.
    num_workers : int, default: 0
        The number of workers for data loading.
    prefetch_factor : int, optional
        The number of batches loaded in advance by each worker.
        Only used when :code:`num_workers > 0`.
    pbar : bool, default: True
        If the progress bar should be shown.

    Returns
    -------
    None
        The virtual stain image is added to the :bdg-danger:`images` slot
        of the WSIData object under the key ``'{model_name}_prediction'``,
        or one image per stain for a virtual stain model.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.tl.virtual_stain(wsi)
        >>> wsi.images["rosie_prediction"]

    """
    import torch
    from lazyslide_models import (
        MODEL_REGISTRY,
        DensePredictionModelProtocol,
        TilePredictionModelProtocol,
        VirtualStainModelProtocol,
    )
    from torch.utils.data import DataLoader

    device = _api.default_value("device", device)
    tile_spec = wsi.tile_spec(tile_key)

    # Resolve model name vs model instance
    if isinstance(model, str):
        model_name = model
        staining_model = MODEL_REGISTRY[model]()
    else:
        staining_model = model
        model_name = getattr(model, "name", model.__class__.__name__).lower()

    # Dispatch on what the model returns, never on its name.
    dense = isinstance(staining_model, DensePredictionModelProtocol)
    if not dense and not isinstance(staining_model, TilePredictionModelProtocol):
        raise TypeError(
            f"virtual_stain needs a tile prediction, marker map or virtual stain "
            f"model, got {type(staining_model).__name__}."
        )

    # Decided once here; everything below follows from these two answers.
    is_stain = isinstance(staining_model, VirtualStainModelProtocol)
    if dense:
        n_channels = (
            3 * len(staining_model.stains)
            if is_stain
            else len(staining_model.channel_names)
        )

    if image_key is None:
        image_key = f"{model_name}_prediction"

    staining_model.to(device)
    staining_model = _api.maybe_compile(staining_model, compile, compile_kws)

    ds = wsi.ds.tile_images(transform=staining_model.get_transform(), tile_key=tile_key)
    loader_kws = _api.loader_kws(device, num_workers, prefetch_factor)
    non_blocking = loader_kws["pin_memory"]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kws)

    with tempfile.TemporaryDirectory() as tmpdir:
        with default_pbar(disable=not pbar) as progress_bar:
            task = progress_bar.add_task("Creating new stains", total=len(ds))

            def predictions():
                amp_ctx = _api.autocast(device, amp, autocast_dtype)
                with amp_ctx, torch.inference_mode():
                    for batch in dl:
                        image = batch["image"].to(device, non_blocking=non_blocking)
                        yield batch, staining_model.predict(image)
                        progress_bar.update(task, advance=len(batch["image"]))

            if dense:
                image, scale = _stitch_dense(
                    wsi, tile_spec, staining_model, predictions(), tmpdir, n_channels
                )
                channel_names = None if is_stain else list(staining_model.channel_names)
            else:
                image, channel_names, scale = _render_per_tile(
                    wsi, tile_spec, staining_model, predictions()
                )
            progress_bar.refresh()

        # The memmap backing a dense image lives in tmpdir, so store it here.
        transform = {"global": Scale(list(scale), axes=("y", "x"))}
        if is_stain:
            # RGB-major: channels 3i to 3i + 2 are stains[i].
            for i, stain in enumerate(staining_model.stains):
                wsi.images[_image_key(image_key, stain)] = Image2DModel.parse(
                    data=image[..., 3 * i : 3 * i + 3].transpose(2, 0, 1),
                    dims=["c", "y", "x"],
                    c_coords=["r", "g", "b"],
                    transformations=transform,
                )
        else:
            wsi.images[image_key] = Image2DModel.parse(
                data=image.transpose(2, 0, 1),
                dims=["c", "y", "x"],
                c_coords=channel_names,
                transformations=transform,
            )


def _tile_grid_index(ys, xs, tile_spec):
    """The pixel each tile maps to, in an image with one pixel per tile.

    Tiles sit exactly one stride apart, so integer division by the stride gives
    every tile a distinct pixel. Scaling by ``(H // stride) / H`` instead, as
    this used to, is slightly less than ``1 / stride``; the rounding error
    builds up across the slide and puts some tiles on the same pixel.
    """
    ys, xs = np.asarray(ys), np.asarray(xs)
    return ys // tile_spec.base_stride_height, xs // tile_spec.base_stride_width


def _render_per_tile(wsi, tile_spec, model, predictions):
    """One pixel per tile. Only ROSIE's output is post-processed."""
    sy, sx = tile_spec.base_stride_height, tile_spec.base_stride_width
    height, width = wsi.properties.shape[:2]
    grid = (-(-height // sy), -(-width // sx))  # ceil division

    image = channel_names = None
    rows, cols = [], []
    for batch, out in predictions:
        if image is None:
            channel_names = list(model.columns or out)
            for name in channel_names:
                kind = np.asarray(out[name]).dtype.kind
                if kind not in "biuf":
                    raise TypeError(
                        f"virtual_stain renders per-tile values as an image, so "
                        f"every column must be numeric; column {name!r} has "
                        f"dtype kind {kind!r}."
                    )
            image = np.zeros((*grid, len(channel_names)), dtype=np.float32)

        values = np.stack(
            [np.asarray(out[n], dtype=np.float32) for n in channel_names], -1
        )
        r, c = _tile_grid_index(batch["y"], batch["x"], tile_spec)
        image[r, c] = values
        rows.extend(r.tolist())
        cols.extend(c.tolist())

    from lazyslide_models.style_transfer import ROSIE

    # Other tile models keep the values they predicted.
    if isinstance(model, ROSIE):
        image = _rosie_postprocess(image, rows, cols)
    return image, channel_names, (sy, sx)


def _rosie_postprocess(image, rows, cols):
    """ROSIE's display post-processing, taken from the ROSIE codebase.

    Clips each channel to its 1st and 99.9th percentile over the tissue,
    stretches it into uint8, then median-blurs it. This belongs to ROSIE's
    output alone: applied to another model it would replace the predicted
    values with a per-slide contrast stretch.
    """
    content = image[rows, cols]
    bg_threshold = np.percentile(content, 1, axis=0)
    max_threshold = np.percentile(content, 99.9, axis=0)
    bg_threshold = np.where(max_threshold > bg_threshold, 0, bg_threshold)
    spread = max_threshold - bg_threshold
    # A constant channel has no spread; leave it at zero instead of dividing by it.
    spread = np.where(spread > 0, spread, 1)
    content = np.clip(content, bg_threshold, max_threshold)
    image[rows, cols] = (content - bg_threshold) * 255.0 / spread
    image = image.astype(np.uint8)
    for channel in range(image.shape[2]):
        image[:, :, channel] = cv2.medianBlur(image[:, :, channel], 3)
    return image


def _stitch_dense(wsi, tile_spec, model, predictions, tmpdir, n_channels):
    """Stitch per-pixel tile predictions into one image, blending overlaps."""
    height, width = wsi.properties.shape[:2]
    if model.output_mpp is None:
        # The output lands on the input tile's own grid.
        scale = 1 / tile_spec.base_downsample
    else:
        if wsi.properties.mpp is None:
            raise ValueError(
                f"{type(model).__name__} sets output_mpp, which needs the slide's "
                f"mpp, but this slide does not record one."
            )
        scale = wsi.properties.mpp / model.output_mpp

    shape = (int(height * scale), int(width * scale))
    image = np.memmap(
        f"{tmpdir}/image.npy",
        dtype=np.float32,
        mode="w+",
        shape=(*shape, n_channels),
    )
    weight = np.zeros(shape, dtype=np.float32)
    mask = None

    for batch, out in predictions:
        # float() first: numpy has no bfloat16, which autocast may produce.
        out = out.detach().float().cpu().numpy()
        if mask is None:
            mask = _blend_mask(*out.shape[-2:])
        for i, tile in enumerate(out):
            y = int(batch["y"][i] * scale)
            x = int(batch["x"][i] * scale)
            y2 = min(y + tile.shape[1], shape[0])
            x2 = min(x + tile.shape[2], shape[1])
            if y2 <= y or x2 <= x:
                continue
            m = mask[: y2 - y, : x2 - x]
            image[y:y2, x:x2] += (
                tile[:, : y2 - y, : x2 - x].transpose(1, 2, 0) * m[..., None]
            )
            weight[y:y2, x:x2] += m

    covered = weight > 0
    image[covered] /= weight[covered][:, np.newaxis]
    return image, (1 / scale, 1 / scale)


def _blend_mask(height, width):
    """Weights that fall off linearly towards the tile edges, for blending."""
    mask = np.ones((height, width), dtype=np.float32)
    ramp_size = int(min(height, width) * 0.1)
    if ramp_size > 0:
        ramp = np.linspace(0.1, 1, ramp_size)
        mask[:ramp_size, :] *= ramp[:, np.newaxis]
        mask[-ramp_size:, :] *= ramp[::-1, np.newaxis]
        mask[:, :ramp_size] *= ramp[np.newaxis, :]
        mask[:, -ramp_size:] *= ramp[np.newaxis, ::-1]
    return mask


def _image_key(prefix, stain):
    """A spatialdata-safe element name; it rejects spaces such as in 'HER2 IHC'."""
    return f"{prefix}_{re.sub(r'[^0-9A-Za-z_.-]+', '_', stain)}"
