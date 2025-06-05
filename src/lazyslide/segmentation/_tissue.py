from __future__ import annotations

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
from lazyslide.models.segmentation import GrandQCTissue


def tissue(
    wsi: WSIData,
    level: int = None,
    bbox_ratio: float = 0.05,
    device: str | None = None,
    key_added: str = Key.tissue,
):
    """
    Return a dataset for tissue segmentation.

    Parameters
    ----------
    wsi: :class:`wsidata.WSIData`
        The whole slide image.
    level : int, default: None
        The level to segment the tissue.
    bbox_ratio : float, default: 0.05
        The ratio of the bounding box to filter
        the false positive tissue polygons.
    device : str, default: None
        The device to run the model.
    key_added : str, default: 'tissues'
        The key to add the tissue polygons.

    """

    if device is None:
        device = get_torch_device()

    props = wsi.properties
    if level is None:
        level_mpp = np.array(props.level_downsample) * props.mpp
        # Get the nearest level that towards mpp=10
        level = np.argmin(np.abs(level_mpp - 10))

    current_mpp = props.level_downsample[level] * props.mpp
    # If reach the mpp=10, we can use the model directly,
    # Otherwise, we need to downsample the image
    if current_mpp < 10:
        scale_factor = 10 / current_mpp
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
    # Ensure the image size is a multiple of 32
    new_height = (height + 31) // 32 * 32
    new_width = (width + 31) // 32 * 32

    # Load the model
    model = GrandQCTissue()
    transform = model.get_transform()
    model.to(device)

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

    pred = pred.squeeze().detach().cpu().numpy()
    mask = np.argmax(pred, axis=0).astype(np.uint8)
    # Flip the mask
    mask = 1 - mask
    polygons = BinaryMask(mask).to_polygons(
        min_area=1e-3,
        min_hole_area=1e-5,
        detect_holes=True,
    )
    polygons["geometry"] = (
        polygons["geometry"]
        # Translate the polygons to the image coordinates
        # Account for the padding
        .translate(xoff=-left_pad, yoff=-top_pad)
        # Scale the polygons to the original image coordinates
        .scale(xfact=current_downsample, yfact=current_downsample, origin=(0, 0))
    )
    minx, miny, width, height = wsi.properties.bounds
    filter_box = scale(
        box(minx, miny, minx + width, miny + height),
        xfact=1 - bbox_ratio,
        yfact=1 - bbox_ratio,
    )
    # Filter polygons that are outside the filter box
    polygons = polygons[polygons.geometry.intersects(filter_box)]
    add_tissues(wsi, key_added, polygons.geometry)
