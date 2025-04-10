from __future__ import annotations

import warnings
from functools import partial
from typing import Literal, Callable, Dict, Mapping

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from shapely.affinity import scale, translate
from torch.utils.data import DataLoader
from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide._utils import default_pbar, get_torch_device
from lazyslide.cv import merge_polygons
from lazyslide.models.base import SegmentationModel
from lazyslide.models.segmentation import Instanseg


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
    runner = SegmentationRunner(
        wsi=wsi,
        model=model,
        tile_key=tile_key,
        transform=transform,
        batch_size=batch_size,
        n_workers=n_workers,
        device=device,
    )
    runner.run()


class SegmentationRunner:
    def __init__(
        self,
        wsi: WSIData,
        model: SegmentationModel,
        tile_key: str,
        transform: Callable = None,
        batch_size: int = 16,
        n_workers: int = 4,
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
        dataloader_kws.setdefault("num_workers", n_workers)
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
