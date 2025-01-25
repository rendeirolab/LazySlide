from __future__ import annotations

import geopandas as gpd
import numpy as np
from lazyslide._utils import default_pbar
from lazyslide.models.base import SegmentationModel
from lazyslide.models.segmentation import Instanseg
from shapely.affinity import translate, scale
from torch.utils.data import DataLoader
from wsidata.io import add_shapes

from lazyslide.cv import PolygonMerger


def cell_segmentation(
    wsi,
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

    # Load the model
    if model == "instanseg":
        model = Instanseg()
        transform = model.get_transform()
        postprocess_fn = model.get_postprocess_fn()
        # TODO: check for tile spec

    dataset = wsi.ds.tile_images(tile_key=tile_key, transform=transform)
    dl = DataLoader(dataset, num_workers=n_workers, batch_size=batch_size)

    # Move model to device
    if device is not None:
        model.to(device)

    downsample = wsi.tile_spec(tile_key).base_downsample

    with default_pbar() as progress_bar:
        task = progress_bar.add_task("Cell segmentation", total=len(dataset))

        polygons = []
        for chunk in dl:
            images = chunk["image"]
            xs, ys = np.asarray(chunk["x"]), np.asarray(chunk["y"])
            if device is not None:
                images = images.to(device)
            output = model.segment(images)

            polys = batch_postprocess(output, xs, ys, postprocess_fn, downsample)
            polygons.extend(polys)
            progress_bar.update(task, advance=len(xs))

        progress_bar.update(task, description="Merging tiles...")
        # === Merge the polygons ===
        merger = PolygonMerger(polygons)
        merger.merge()
        # === Refresh the progress bar ===
        progress_bar.update(task, description="Cell segmentation")
        progress_bar.refresh()

    cells_df = (
        gpd.GeoDataFrame({"geometry": merger.merged_polygons})
        .explode()
        .reset_index(drop=True)
    )
    add_shapes(wsi, key_added, cells_df)


def batch_postprocess(output, xs, ys, postprocess_fn, downsample):
    polygons = []
    for img, x, y in zip(output, xs, ys):
        result = postprocess_fn(img.squeeze(0))
        polys = result["polygons"]
        # transform the polygons to the global coordinate
        for poly in polys:
            poly = scale(poly, xfact=downsample, yfact=downsample, origin=(0, 0))
            poly = translate(poly, xoff=x, yoff=y)
            polygons.append(poly)

    return polygons
