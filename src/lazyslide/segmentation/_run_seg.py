from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import torch
from shapely.affinity import scale, translate
from torch.utils.data import DataLoader
from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide._utils import default_pbar
from lazyslide.cv import merge_polygons
from lazyslide.models.base import SegmentationModel
from lazyslide.models.segmentation import Instanseg


def cells(
    wsi: WSIData,
    model: str | SegmentationModel = "instanseg",
    tile_key="tiles",
    transform=None,
    batch_size=16,
    n_workers=4,
    device=None,
    key_added="cells",
):
    """Cell segmentation for the whole slide image.

    Tiles should be prepared before segmentation.

    Recommended tile setting:
    - **instanseg**: 512x512, mpp=0.5

    Parameters
    ----------
    wsi : WSIData
        The whole slide image data.
    model : str | SegmentationModel, default: "instanseg"
        The cell segmentation model.
    tile_key : str, default: "tiles"
        The key of the tile table.
    transform : callable, default: None
        The transformation for the input tiles.
    batch_size : int, default: 16
        The batch size for segmentation.
    n_workers : int, default: 4
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    key_added : str, default: "cells"
        The key for the added cell shapes.

    """
    if model == "instanseg":
        model = Instanseg()
        # Run tile check
        tile_spec = wsi.tile_spec(tile_key)
        check_mpp = tile_spec.mpp == 0.5
        check_size = tile_spec.height == 512 and tile_spec.width == 512
        if not check_mpp or not check_size:
            warnings.warn(
                f"To optimize the performance of Instanseg model, "
                f"the tile size should be 512x512 and the mpp should be 0.5. "
                f"Current tile size is {tile_spec.width}x{tile_spec.height} with {tile_spec.mpp} mpp."
            )

    return _seg_runner(
        wsi,
        model,
        tile_key,
        transform,
        batch_size,
        n_workers,
        device,
        key_added,
        task="multiclass",
        task_name="Cell Segmentation",
    )


def semantic(
    wsi: WSIData,
    model: SegmentationModel,
    tile_key="tiles",
    transform=None,
    batch_size=16,
    n_workers=4,
    device=None,
    key_added="anatomical_structures",
):
    """
    Semantic segmentation for the whole slide image.

    Parameters
    ----------
    wsi : WSIData
        The whole slide image data.
    model : SegmentationModel
        The segmentation model.
    tile_key : str, default: "tiles"
        The key of the tile table.
    transform : callable, default: None
        The transformation for the input tiles.
    batch_size : int, default: 16
        The batch size for segmentation.
    n_workers : int, default: 4
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    key_added : str, default: "anatomical_structures"
        The key for the added instance shapes.

    """
    return _seg_runner(
        wsi,
        model,
        tile_key,
        transform,
        batch_size,
        n_workers,
        device,
        key_added,
        task="multilabel",
        task_name="Semantic Segmentation",
    )


def _seg_runner(
    wsi: WSIData,
    model: SegmentationModel,
    tile_key="tiles",
    transform=None,
    batch_size=16,
    n_workers=4,
    device=None,
    key_added="segmentation",
    task: Literal["multilabel", "multiclass"] = "multilabel",
    task_name: str = "Segmentation",
):
    if transform is None:
        transform = model.get_transform()
    postprocess_fn = model.get_postprocess()

    dataset = wsi.ds.tile_images(tile_key=tile_key, transform=transform)
    dl = DataLoader(dataset, num_workers=n_workers, batch_size=batch_size)

    # Move model to device
    if device is not None:
        model.to(device)

    downsample = wsi.tile_spec(tile_key).base_downsample

    with default_pbar() as progress_bar:
        task = progress_bar.add_task(task_name, total=len(dataset))

        results = []
        for chunk in dl:
            images = chunk["image"]
            xs, ys = np.asarray(chunk["x"]), np.asarray(chunk["y"])
            if device is not None:
                images = images.to(device)
            output = model.segment(images)

            if isinstance(output, torch.Tensor):
                # If the output is a tensor, convert it to numpy
                output = output.cpu().detach().numpy()

            rs = _batch_postprocess(output, xs, ys, postprocess_fn, downsample)
            results.extend(rs)
            progress_bar.update(task, advance=len(xs))

        polys_df = pd.concat(results).reset_index(drop=True)
        progress_bar.update(task, description="Merging tiles...")
        # === Merge the polygons ===
        polys_df = merge_polygons(polys_df)
        # === Refresh the progress bar ===
        progress_bar.update(task, description=task_name)
        progress_bar.refresh()

    polys_df = polys_df.explode().reset_index(drop=True)
    add_shapes(wsi, key_added, polys_df)


def _batch_postprocess(output: np.ndarray, xs, ys, postprocess_fn, downsample):
    results = []
    for img, x, y in zip(output, xs, ys):
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = postprocess_fn(img)
        # The output of postprocess_fn is a gpd.GeoDataFrame
        # transform the polygons to the global coordinate
        polys = []
        for poly in result["geometry"]:
            poly = scale(poly, xfact=downsample, yfact=downsample, origin=(0, 0))
            poly = translate(poly, xoff=x, yoff=y)
            polys.append(poly)
        result["geometry"] = polys
        results.append(result)

    return results
