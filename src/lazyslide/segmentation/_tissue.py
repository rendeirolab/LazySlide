from __future__ import annotations

import warnings
from typing import Literal

import cv2
import numpy as np
import torch
from shapely import box
from shapely.affinity import scale
from wsidata import WSIData
from wsidata.io import add_tissues

from lazyslide._const import Key
from lazyslide._utils import get_torch_device
from lazyslide.cv import BinaryMask
from lazyslide.models.segmentation import GrandQCTissue, PathProfilerTissueSegmentation


def tissue(
    wsi: WSIData,
    *,
    model: Literal["grandqc", "pathprofiler"] = "pathprofiler",
    level: int = None,
    bbox_ratio: float = 0.05,
    min_area=1e-3,
    min_hole_area=1e-5,
    detect_holes: bool = True,
    threshold: float = 0.5,
    device: str | None = None,
    key_added: str = Key.tissue,
):
    """
    Perform tissue segmentation powered by a deep learning model.

    Supported models:
        - "grandqc":
          :class:`GrandQCTissue <lazyslide.models.segmentation.GrandQCTissue>`,
          :cite:p:`Weng2024-jf`.
        - "pathprofiler":
          :class:`PathProfilerTissueSegmentation <lazyslide.models.segmentation.PathProfilerTissueSegmentation>`,
          :cite:p:`Haghighat2022-sy`.

    Parameters
    ----------
    wsi : :class:`wsidata.WSIData`
        The whole slide image.
    model : {"grandqc", "pathprofiler"}, default: "pathprofiler"
        The model to use for tissue segmentation.
    level : int, default: None
        The level to segment the tissue.
    bbox_ratio : float, default: 0.05
        The ratio of the bounding box to filter
        the false positive tissue polygons.
    min_area : float, default: 1e-3
        The minimum area of the tissue polygon.
    min_hole_area : float, default: 1e-5
        The minimum area of the hole in the tissue polygon.
    detect_holes : bool, default: True
        Whether to detect holes in the tissue polygons.
    threshold : float, default: 0.5
        The probability threshold to consider a pixel as tissue.
    device : str, default: None
        The device to run the model.
    key_added : str, default: 'tissues'
        The key to add the tissue polygons.

    """

    if device is None:
        device = get_torch_device()

    # Load the model
    model_name = model
    if model == "grandqc":
        model = GrandQCTissue()
        target_mpp = 10
        min_size = 32
        divider = 32
    elif model == "pathprofiler":
        model = PathProfilerTissueSegmentation()
        target_mpp = 2.5
        divider = 64
        min_size = 128
    else:
        raise ValueError(
            f"Unknown model: {model}, choose from 'grandqc' or 'pathprofiler'."
        )
    transform = model.get_transform()
    model.to(device)

    props = wsi.properties
    if level is None:
        level_mpp = np.array(props.level_downsample) * props.mpp
        # Get the nearest level that towards target mpp
        level = np.argmin(np.abs(level_mpp - target_mpp))

    current_mpp = props.level_downsample[level] * props.mpp
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

    # Simulate JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    result, img = cv2.imencode(".jpg", img, encode_param)
    img = cv2.imdecode(img, 1)

    img = torch.tensor(img).permute(2, 0, 1)

    img_t = transform(img).unsqueeze(0)
    img_t = img_t.to(device)
    pred = model.segment(img_t)
    pred = pred["probability_map"]

    pred = pred.squeeze(0).detach().cpu().numpy()
    if model_name == "grandqc":
        tissue_prob = pred[0]
    else:
        tissue_prob = pred[1]
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
    if len(polygons) == 0:
        warnings.warn("No tissues were found. The staining might be too weak.")
        return
    add_tissues(wsi, key_added, polygons.geometry)
