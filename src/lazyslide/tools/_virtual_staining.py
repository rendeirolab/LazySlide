import tempfile
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale
from torch.utils.data import DataLoader
from wsidata import WSIData

from lazyslide._const import Key
from lazyslide._utils import default_pbar, get_torch_device
from lazyslide.models import MODEL_REGISTRY


def virtual_stain(
    wsi: WSIData,
    model: str = "rosie",
    image_key: str = None,
    tile_key: str = Key.tiles,
    device: str = None,
    amp: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
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
    tile_spec = wsi.tile_spec(tile_key)
    if model == "rosie":
        image_shape = (
            wsi.properties.shape[0] // tile_spec.base_stride_height,
            wsi.properties.shape[1] // tile_spec.base_stride_width,
            50,
        )
        scale_x = image_shape[1] / wsi.properties.shape[1]
        scale_y = image_shape[0] / wsi.properties.shape[0]
    else:
        raise ValueError(f"Model {model} not supported.")

    if image_key is None:
        image_key = f"{model}_prediction"

    if device is None:
        device = get_torch_device()

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
            task = progress_bar.add_task("Extracting features", total=len(ds))

            if isinstance(device, torch.device):
                device = device.type
            amp_ctx = torch.autocast(device, autocast_dtype) if amp else nullcontext()
            with amp_ctx, torch.inference_mode():
                for batch in dl:
                    expression = staining_model.predict(batch["image"].to(device))
                    image_x = (batch["x"] * scale_x).long() + 1
                    image_y = (batch["y"] * scale_y).long() + 1
                    expression = expression.detach().cpu().numpy()

                    mask_x.extend(image_x.tolist())
                    mask_y.extend(image_y.tolist())

                    new_image[image_y, image_x] = expression
                    progress_bar.update(task, advance=len(batch["image"]))
            progress_bar.refresh()

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
