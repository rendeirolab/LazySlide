from __future__ import annotations

from functools import partial
from typing import Literal, Callable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely.affinity import scale, translate
from torch.utils.data import DataLoader
from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide._const import Key
from lazyslide._utils import default_pbar, get_torch_device
from lazyslide.cv import merge_polygons
from lazyslide.models.base import SegmentationModel


def semantic(
    wsi: WSIData,
    model: SegmentationModel,
    tile_key=Key.tiles,
    transform=None,
    batch_size=4,
    num_workers=0,
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
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    key_added : str, default: "anatomical_structures"
        The key for the added instance shapes.

    """
    runner = SegmentationRunner(
        wsi=wsi,
        model=model,
        tile_key=tile_key,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    shapes = runner.run()
    # Add the segmentation results to the WSIData
    add_shapes(wsi, key=key_added, shapes=shapes)


class SegmentationRunner:
    """
    Segmentation runner for the whole slide image.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole slide image data.
    model : :class:`SegmentationModel <lazyslide.models.base.SegmentationModel>`
        The segmentation model.
    tile_key : str
        The key of the tile table.
    transform : callable, default: None
        The transformation for the input tiles.
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    postprocess_kws : dict, default: None
        The keyword arguments for the postprocess function defined in the model class
    dataloader_kws : dict, default: None
        The keyword arguments for the DataLoader.
    class_col : str, default: None
        The column name for the class in the output GeoDataFrame.
    prob_col : str, default: None
        The column name for the probability in the output GeoDataFrame.
    buffer_px : int, default: 0
        The buffer size in pixels for the polygons.
    drop_overlap : float, default: 0.9
        The overlap threshold for dropping polygons.
    pbar : bool, default: True
        Whether to show the progress bar.

    """

    def __init__(
        self,
        wsi: WSIData,
        model: SegmentationModel,
        tile_key: str,
        transform: Callable = None,
        batch_size: int = 4,
        num_workers: int = 0,
        device: str = None,
        postprocess_kws: dict = None,
        dataloader_kws: dict = None,
        class_col: str = None,
        prob_col: str = None,
        buffer_px: int = 0,
        drop_overlap: float = 0.9,
        pbar: bool = True,
    ):
        self.wsi = wsi
        self.model = model
        if device is None:
            device = get_torch_device()
        self.device = device
        self.tile_key = tile_key
        self.downsample = wsi.tile_spec(tile_key).base_downsample

        if transform is None:
            transform = model.get_transform()
        self.transform = transform

        if postprocess_kws is None:
            postprocess_kws = {}
        postprocess_fn = model.get_postprocess()
        self.postprocess_fn = partial(postprocess_fn, **postprocess_kws)

        if dataloader_kws is None:
            dataloader_kws = {}
        dataloader_kws.setdefault("num_workers", num_workers)
        dataloader_kws.setdefault("batch_size", batch_size)
        self.dataloader_kws = dataloader_kws
        self.merge_kws = dict(
            class_col=class_col,
            prob_col=prob_col,
            buffer_px=buffer_px,
            drop_overlap=drop_overlap,
        )

        self.pbar = pbar

    def _batch_postprocess(self, output, xs, ys):
        results = []

        if isinstance(output, (torch.Tensor, np.ndarray)):
            batches = zip(output, xs, ys)
        elif isinstance(output, tuple):
            batches = zip(list(zip(*output)), xs, ys)
        elif isinstance(output, Mapping):
            flattened = [
                dict(zip(output.keys(), values)) for values in zip(*output.values())
            ]
            batches = zip(flattened, xs, ys)
        else:
            raise NotImplementedError(f"Unsupported model output type {type(output)}")

        for batch, x, y in batches:
            result = self.postprocess_fn(batch)
            # The output of postprocess_fn is a gpd.GeoDataFrame
            # transform the polygons to the global coordinate
            polys = []
            for poly in result["geometry"]:
                poly = scale(
                    poly, xfact=self.downsample, yfact=self.downsample, origin=(0, 0)
                )
                poly = translate(poly, xoff=x, yoff=y)
                polys.append(poly)
            result["geometry"] = polys
            if len(result) > 0:
                results.append(result)

        return results

    def __call__(self):
        dataset = self.wsi.ds.tile_images(
            tile_key=self.tile_key, transform=self.transform
        )
        dl = DataLoader(dataset, **self.dataloader_kws)

        # Move model to device
        if self.device is not None:
            self.model.to(self.device)

        with default_pbar(disable=not self.pbar) as progress_bar:
            task = progress_bar.add_task("Segmentation", total=len(dataset))

            results = []
            for chunk in dl:
                images = chunk["image"]
                xs, ys = np.asarray(chunk["x"]), np.asarray(chunk["y"])
                if self.device is not None:
                    images = images.to(self.device)
                output = self.model.segment(images)

                rs = self._batch_postprocess(output, xs, ys)
                # Update only if the output is not empty
                results.extend(rs)
                progress_bar.update(task, advance=len(xs))
            polys_df = gpd.GeoDataFrame(pd.concat(results).reset_index(drop=True))
            progress_bar.update(task, description="Merging tiles...")
            # === Merge the polygons ===
            polys_df = merge_polygons(polys_df, **self.merge_kws)
            # === Refresh the progress bar ===
            progress_bar.update(task, description="Segmentation")
            progress_bar.refresh()

        polys_df = polys_df.explode().reset_index(drop=True)
        return polys_df

    def run(self):
        """
        Run the segmentation.
        """
        return self.__call__()
