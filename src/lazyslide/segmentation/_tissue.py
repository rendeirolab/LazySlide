from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cv2
import numpy as np
from shapely import box
from shapely.affinity import scale
from wsidata import WSIData
from wsidata.io import add_tissues

from lazyslide import _api
from lazyslide._const import Key
from lazyslide.cv import BinaryMask
from lazyslide.cv.mask import repair_invalid_geometry

if TYPE_CHECKING:
    import torch
    from lazyslide_models import SegmentationModelProtocol

# Short names for the built-in tissue models, besides their registry keys
_TISSUE_MODEL_ALIASES = {
    "grandqc": "grandqc-tissue",
    "hest": "hest-tissue-segmentation",
}


def _tissue_model_spec(model) -> tuple[float | None, int, int, int, bool]:
    """Return ``(target_mpp, divider, min_size, tissue_channel, jpeg_roundtrip)``.

    ``target_mpp`` is None for a model whose input resolution is unknown.
    """
    # ponytail: the models don't publish their input mpp yet, so it's kept here per
    # model class; read it from InputConstraint once
    # rendeirolab/lazyslide-models#37 lands.
    from lazyslide_models.segmentation import (
        GrandQCTissue,
        HESTTissueSegmentation,
        PathProfilerTissueSegmentation,
    )

    if isinstance(model, GrandQCTissue):
        # GrandQC's tissue detector was trained on JPEG-compressed images, so
        # upstream re-encodes at quality 80; its tissue is channel 0
        return 10, 32, 32, 0, True
    if isinstance(model, PathProfilerTissueSegmentation):
        # Upstream's --mask_magnification is 1.25x or 2.5x, i.e. ~8 or ~4 µm/px
        return 4, 64, 128, 1, False
    if isinstance(model, HESTTissueSegmentation):
        return 1, 8, 8, 1, False
    constraint = getattr(model, "input_constraint", None)
    divider = getattr(constraint, "divisible_by", None) or 1
    min_size = getattr(constraint, "min", None) or 1
    classes = getattr(model, "classes", None) or ()
    channel = list(classes).index("Tissue") if "Tissue" in classes else 1
    return None, divider, min_size, channel, False


def _tile_starts(size: int, tile: int) -> list[int]:
    """Offsets of ``tile`` px windows covering ``size`` px.

    Windows overlap by a quarter tile; the last one is flush with the end.
    """
    if size <= tile:
        return [0]
    return [*range(0, size - tile, tile - tile // 4), size - tile]


def _split_transform(transform) -> tuple[list, list]:
    """Split ``transform`` into steps for the whole image and steps per tile.

    The per-tile steps are its trailing per-pixel ones (ToImage, ToDtype,
    Normalize), which give the same pixels on a tile as on the whole image.
    The rest, like PathProfiler's image-global CLAHE, runs on the whole image,
    as does a transform that is not a Compose.
    """
    from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

    steps = transform.transforms if isinstance(transform, Compose) else [transform]
    n = len(steps)
    while n and isinstance(steps[n - 1], (ToImage, ToDtype, Normalize)):
        n -= 1
    return steps[:n], steps[n:]


def _segment_tiled(
    model, img, tile_steps, device, tissue_class: int, tile_px: int
) -> np.ndarray:
    """Tissue probability map of ``img`` ([C, H, W]) from overlapping tiles.

    ``tile_steps`` finish the model transform tile by tile, so only one tile at
    a time is float. Overlaps are blended with a Gaussian window, so each pixel
    is decided mostly by the tile it sits most central in.
    """
    from ._seg_runner import create_importance_map

    height, width = img.shape[-2:]
    # tile_px and the padded image are both multiples of the model's divider,
    # so every tile side is too
    tile_h, tile_w = min(tile_px, height), min(tile_px, width)
    weight = create_importance_map((tile_h, tile_w)).numpy()
    prob = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for y in _tile_starts(height, tile_h):
        for x in _tile_starts(width, tile_w):
            tile = img[..., y : y + tile_h, x : x + tile_w]
            for step in tile_steps:
                tile = step(tile)
            tile = tile.unsqueeze(0).to(device)
            pred = model.segment(tile).probability_map[0, tissue_class]
            window = np.s_[y : y + tile_h, x : x + tile_w]
            prob[window] += pred.float().cpu().numpy() * weight
            weight_sum[window] += weight
    prob /= weight_sum
    return prob


def tissue(
    wsi: WSIData,
    *,
    model: str | SegmentationModelProtocol = "pathprofiler",
    level: int | None = None,
    mpp: float | None = None,
    bbox_ratio: float = 0.05,
    min_area=1e-3,
    min_hole_area=1e-5,
    detect_holes: bool = True,
    threshold: float = 0.5,
    tile_px: int = 1024,
    device: str | None = None,
    amp: bool | None = None,
    autocast_dtype: torch.dtype = None,
    compile: bool | None = None,
    compile_kws: dict | None = None,
    key_added: str = Key.tissue,
):
    """
    Perform :term:`tissue segmentation` powered by a deep learning model.

    Models with a built-in input resolution:
        - "pathprofiler": :cite:p:`Haghighat2022-sy`. Runs on mpp=4 (2.5x).
        - "grandqc" (registry key "grandqc-tissue"): :cite:p:`Weng2024-jf`. Runs on mpp=10.
        - "hest" (registry key "hest-tissue-segmentation"):
          "https://huggingface.co/MahmoodLab/hest-tissue-seg". Runs on mpp=1.

    Any other segmentation model runs at the ``mpp`` or ``level`` you pass.

    The model runs on overlapping tiles at the working :term:`mpp` (see
    ``tile_px``), blended with a Gaussian window, so its memory use does not
    grow with the slide: HEST at 1 µm/px on a 20k px slide is a ~10k px image,
    far too big for one forward pass. An image that fits in one tile goes
    through in a single pass.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The :term:`whole slide image <WSI>`.
    model : str or SegmentationModelProtocol, default: "pathprofiler"
        The model to use for :term:`tissue segmentation`: a model registry key
        (see :ref:`models-section`) or a model instance.
    level : int, default: None
        The level to segment the tissue, mutually exclusive with mpp.
    mpp : float, default: None
        The mpp level to segment the tissue, mutually exclusive with level.
    bbox_ratio : float, default: 0.05
        The ratio of the bounding box to filter
        the false positive tissue :term:`polygons <polygon>`.
    min_area : float, default: 1e-3
        The minimum area of the tissue polygon.
    min_hole_area : float, default: 1e-5
        The minimum area of the hole in the tissue polygon.
    detect_holes : bool, default: True
        Whether to detect :term:`holes` in the tissue polygons.
    threshold : float, default: 0.5
        The probability threshold to consider a pixel as tissue.
    tile_px : int, default: 1024
        The tile size in px at the working :term:`mpp`, rounded down to a size
        the model accepts. Smaller tiles save memory; bigger ones give the model
        more context.
    device : str, default: None
        The device to run the model.
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
    key_added : str, default: 'tissues'
        The key to add the tissue polygons.

    Returns
    -------
    None
        The tissue polygons are added to the :bdg-danger:`shapes` slot
        of the WSIData object.

    """
    import torch

    device = _api.default_value("device", device)

    # Load the model
    if isinstance(model, str):
        from lazyslide_models import MODEL_REGISTRY

        model = MODEL_REGISTRY[_TISSUE_MODEL_ALIASES.get(model, model)]()
    target_mpp, divider, min_size, tissue_class, jpeg_roundtrip = _tissue_model_spec(
        model
    )
    if target_mpp is None and mpp is None and level is None:
        raise ValueError(
            f"The input resolution of {type(model).__name__} is unknown; "
            "pass `mpp` or `level`."
        )
    if tile_px < 1:
        raise ValueError(f"tile_px must be a positive number of px, got {tile_px}.")
    # Tiles must be multiples of the divider, like the padded image
    tile_px = max(min_size, tile_px // divider * divider)
    whole_steps, tile_steps = _split_transform(model.get_transform())
    model.to(device)
    model = _api.maybe_compile(model, compile, compile_kws)

    props = wsi.properties
    if mpp is not None and level is not None:
        raise ValueError("Please specify either level or mpp, not both.")
    if mpp is not None:
        target_mpp = mpp
    if level is None:
        level_mpp = np.array(props.level_downsample) * props.mpp
        # Get the nearest level that towards target mpp
        level = np.argmin(np.abs(level_mpp - target_mpp))

    current_mpp = props.level_downsample[level] * props.mpp
    if target_mpp is None:
        # A model without a known resolution runs at the level it was given
        target_mpp = current_mpp
    # If reach the target mpp, we can use the model directly,
    # Otherwise, we need to downsample the image
    if current_mpp < target_mpp:
        scale_factor = target_mpp / current_mpp
    else:
        scale_factor = 1

    # Get the tissue image
    height, width = props.level_shape[level]
    img = wsi.reader.get_region(0, 0, width, height, level=level)
    # Downsample the image if necessary
    if scale_factor != 1:
        t_width = int(width / scale_factor)
        t_height = int(height / scale_factor)
        # Update the scale factor to avoid errors due to rounding
        scale_factor = width / t_width
        img = cv2.resize(
            img,
            (t_width, t_height),
            interpolation=cv2.INTER_LINEAR,
        )
    current_downsample = props.level_downsample[level] * scale_factor
    height, width = img.shape[:2]
    # Ensure the image size is a multiple of divider
    new_height = max(min_size, (height + divider - 1) // divider * divider)
    new_width = max(min_size, (width + divider - 1) // divider * divider)

    # We cannot read the image directly from the reader.
    # The padding from image reader will introduce padding at only two sides
    # We need to pad the image on all four sides
    # without shifting the image equilibrium
    # Otherwise, this will introduce artifacts in the segmentation

    # # Compute padding amounts
    top_pad = (new_height - height) // 2
    bottom_pad = new_height - height - top_pad
    left_pad = (new_width - width) // 2
    right_pad = new_width - width - left_pad

    # Apply padding
    img_height, img_width = img.shape[:2]
    img = np.pad(
        img,
        pad_width=((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
        mode="constant",
        constant_values=0,  # Pad with black pixels
    )

    if jpeg_roundtrip:
        # The round trip keeps the reader's RGB order: imdecode returns
        # channels in the order imencode was given them.
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _result, img = cv2.imencode(".jpg", img, encode_param)
        img = cv2.imdecode(img, 1)

    # A uint8 view, not a copy: only image-global transform steps, like
    # PathProfiler's CLAHE, see the whole image; the per-pixel rest runs per tile
    img = torch.from_numpy(img).permute(2, 0, 1)
    for step in whole_steps:
        img = step(img)
    amp_ctx = _api.autocast(device, amp, autocast_dtype)
    with amp_ctx, torch.inference_mode():
        tissue_prob = _segment_tiled(
            model, img, tile_steps, device, tissue_class, tile_px
        )
    mask = (tissue_prob > threshold).astype(np.uint8)
    # Unpad the mask to match the original image size
    mask = mask[top_pad : top_pad + img_height, left_pad : left_pad + img_width]
    polygons = BinaryMask(mask).to_polygons(
        min_area=min_area,
        min_hole_area=min_hole_area,
        detect_holes=detect_holes,
    )
    polygons["geometry"] = (
        polygons["geometry"]
        # Scale the polygons to the original image coordinates
        .scale(xfact=current_downsample, yfact=current_downsample, origin=(0, 0))
    )
    minx, miny, width, height = wsi.properties.bounds
    filter_box = scale(
        box(minx, miny, minx + width, miny + height),
        xfact=1 - bbox_ratio,
        yfact=1 - bbox_ratio,
    )
    # Only polygons that are in the filter box are kept
    polygons = polygons[polygons.geometry.intersects(filter_box)]
    # Contours from cv2.findContours can pinch to a point, producing self-touching
    # rings that GEOS rejects. Repair here so invalid geometry never reaches the
    # shapes slot and blows up an unrelated set operation later.
    polygons = polygons.assign(geometry=repair_invalid_geometry(polygons.geometry))
    polygons = polygons[~polygons.geometry.is_empty]
    if len(polygons) == 0:
        warnings.warn("No tissues were found. The staining might be too weak.")
        return
    add_tissues(wsi, key_added, polygons.geometry)
