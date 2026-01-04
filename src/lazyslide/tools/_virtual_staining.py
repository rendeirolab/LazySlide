import tempfile
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale
from torch.utils.data import DataLoader
from wsidata import WSIData

from lazyslide import _api
from lazyslide._const import Key
from lazyslide._utils import default_pbar
from lazyslide.models import MODEL_REGISTRY


def virtual_stain(
    wsi: WSIData,
    model: str = "rosie",
    image_key: str = None,
    tile_key: str = Key.tiles,
    device: str = None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pbar: bool = True,
):
    """
    Translate the HE images to multiplexed images.

    A new multi-channel image will be created and stored in the WSIData object.
    The marker name is recorded in the image channel names.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image data to work on.
    model : str, default: "rosie"
        The virtual staining model to use.
    image_key : str, default: None
        The key to store the new image.
    tile_key : str, default: "tiles"
        The key for the tile table.
    device : str, default: None
        Which device to use for inference. If None, the default device is used.
    amp : bool, default: False
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, default: torch.float16
        The dtype for automatic mixed precision.
    batch_size : int, default: 32
        The batch size for inference.
    num_workers : int, default: 0
        The number of workers for data loading.
    pbar : bool, default: True
        If the progress bar should be shown.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.tl.virtual_stain(wsi)
        >>> wsi.images["rosie_prediction"]

    """
    amp = _api.default_value("amp", amp)
    autocast_dtype = _api.default_value("autocast_dtype", autocast_dtype)
    device = _api.default_value("device", device)

    tile_spec = wsi.tile_spec(tile_key)
    if model == "rosie":
        image_shape = (
            wsi.properties.shape[0] // tile_spec.base_stride_height,
            wsi.properties.shape[1] // tile_spec.base_stride_width,
            50,
        )
        scale_x = image_shape[1] / wsi.properties.shape[1]
        scale_y = image_shape[0] / wsi.properties.shape[0]
    elif model == "gigatime":
        # The output of Giga-TIME is the same size as the input image
        img_y, img_x = wsi.properties.shape[:2]
        scale_x = 1 / tile_spec.base_downsample
        scale_y = 1 / tile_spec.base_downsample
        image_shape = (int(img_y * scale_y), int(img_x * scale_x), 23)
    else:
        raise ValueError(f"Model {model} not supported.")

    if image_key is None:
        image_key = f"{model}_prediction"

    with tempfile.TemporaryDirectory() as tmpdir:
        new_image = np.memmap(
            f"{tmpdir}/image.npy", dtype=np.float32, mode="w+", shape=image_shape
        )

        staining_model = MODEL_REGISTRY[model]()
        staining_model.to(device)

        transform = staining_model.get_transform()

        ds = wsi.ds.tile_images(transform=transform, tile_key=tile_key)
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        mask_x, mask_y = [], []

        with default_pbar(disable=not pbar) as progress_bar:
            task = progress_bar.add_task("Creating new stains", total=len(ds))

            if isinstance(device, torch.device):
                device = device.type
            amp_ctx = torch.autocast(device, autocast_dtype) if amp else nullcontext()
            with amp_ctx, torch.inference_mode():
                if model == "rosie":
                    for batch in dl:
                        expression = staining_model.predict(batch["image"].to(device))
                        image_x = (batch["x"] * scale_x).long() + 1
                        image_y = (batch["y"] * scale_y).long() + 1
                        expression = expression.detach().cpu().numpy()

                        mask_x.extend(image_x.tolist())
                        mask_y.extend(image_y.tolist())

                        new_image[image_y, image_x] = expression
                        progress_bar.update(task, advance=len(batch["image"]))
                elif model == "gigatime":
                    weight_image = np.zeros(image_shape[:2], dtype=np.float32)
                    # Create a weight mask for blending
                    _, tile_h, tile_w = ds[0]["image"].shape
                    weight_mask = np.ones((tile_h, tile_w), dtype=np.float32)
                    # Linear ramp for the edges
                    ramp_size = int(min(tile_h, tile_w) * 0.1)
                    if ramp_size > 0:
                        ramp = np.linspace(0.1, 1, ramp_size)
                        weight_mask[:ramp_size, :] *= ramp[:, np.newaxis]
                        weight_mask[-ramp_size:, :] *= ramp[::-1, np.newaxis]
                        weight_mask[:, :ramp_size] *= ramp[np.newaxis, :]
                        weight_mask[:, -ramp_size:] *= ramp[np.newaxis, ::-1]

                    for batch in dl:
                        predicted_channels = staining_model.predict(
                            batch["image"].to(device)
                        )
                        predicted_channels = torch.sigmoid(predicted_channels)
                        predicted_channels = predicted_channels.detach().cpu().numpy()
                        for ix, cs in enumerate(predicted_channels):
                            image_x = (batch["x"][ix] * scale_x).long().item()
                            image_y = (batch["y"][ix] * scale_y).long().item()

                            # Actual tile size might be different from expected if not padded
                            c, th, tw = cs.shape

                            # Clip to image boundaries
                            y1, y2 = image_y, min(image_y + th, image_shape[0])
                            x1, x2 = image_x, min(image_x + tw, image_shape[1])

                            if y2 <= y1 or x2 <= x1:
                                continue

                            tile_slice_y = slice(0, y2 - y1)
                            tile_slice_x = slice(0, x2 - x1)

                            prediction = cs[:, tile_slice_y, tile_slice_x].transpose(
                                1, 2, 0
                            )
                            mask = weight_mask[tile_slice_y, tile_slice_x]

                            new_image[y1:y2, x1:x2] += (
                                prediction * mask[:, :, np.newaxis]
                            )
                            weight_image[y1:y2, x1:x2] += mask

                        progress_bar.update(task, advance=len(batch["image"]))

                    # Normalize by weights
                    nonzero_weight = weight_image > 0
                    new_image[nonzero_weight] /= weight_image[nonzero_weight][
                        :, np.newaxis
                    ]

            progress_bar.refresh()

        # Postprocessing
        if model == "rosie":
            # Apply postprocessing from the ROSIE codebase
            content_region = new_image[mask_y, mask_x]
            bg_threshold = np.percentile(content_region, 1, axis=0)
            max_threshold = np.percentile(content_region, 99.9, axis=0)
            # Set bg_threshold to 0 if max_threshold is greater than bg_threshold
            bg_threshold = np.where(max_threshold > bg_threshold, 0, bg_threshold)
            new_image[mask_y, mask_x] = np.clip(
                new_image[mask_y, mask_x], bg_threshold, max_threshold
            )
            # Normalize to (0, 255)
            new_image[mask_y, mask_x] = (
                (new_image[mask_y, mask_x] - bg_threshold)
                * 255.0
                / (max_threshold - bg_threshold)
            )
            new_image = new_image.astype(np.uint8)
            for channel in range(new_image.shape[2]):
                new_image[:, :, channel] = cv2.medianBlur(new_image[:, :, channel], 3)
        # Write to spatialdata
        image = Image2DModel.parse(
            data=new_image.transpose(2, 0, 1),
            dims=["c", "y", "x"],
            c_coords=staining_model.get_channel_names(),
            transformations={
                "global": Scale([1 / scale_y, 1 / scale_x], axes=("y", "x"))
            },
        )
        wsi.images[image_key] = image
