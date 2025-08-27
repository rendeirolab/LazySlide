from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, List, Literal, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely import box, prepare
from torch.utils.data import DataLoader
from wsidata import TileSpec, WSIData
from wsidata.io import add_shapes

from lazyslide._const import Key
from lazyslide._utils import default_pbar, get_torch_device
from lazyslide.cv import (
    InstanceMap,
    ProbabilityMap,
    merge_connected_polygons,
    nms,
)
from lazyslide.models.base import SegmentationModel


def semantic(
    wsi: WSIData,
    model: SegmentationModel,
    tile_key=Key.tiles,
    class_names: List[str] | Mapping[int, str] | None = None,
    transform=None,
    mode: Literal["constant", "gaussian"] = "gaussian",
    sigma_scale: float = 0.125,
    low_memory: bool = False,
    threshold: float = 0.5,
    ignore_index: int | None = 0,
    buffer_px: int = 2,
    chunk_size: int = 512,
    batch_size=4,
    num_workers=0,
    device=None,
    pbar: bool = True,
    key_added="anatomical_structures",
):
    """
    Semantic segmentation for the whole slide image.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : SegmentationModel
        The segmentation model.
    tile_key : str, default: "tiles"
        The key of the tile table.
    class_names : list of str or dict, optional
        The class names for the segmentation. Either a list of class names or a dict mapping class indices to names.
    transform : callable, default: None
        The transformation to apply to each input tile before segmentation.
    mode : {"constant", "gaussian"}, default: "gaussian"
        The probability distribution to apply for the prediction map.
        "constant" uses uniform weights, "gaussian" applies a Gaussian weighting.
    sigma_scale : float, default: 0.125
        The scale of the Gaussian sigma for the importance map if mode is "gaussian".
    low_memory : bool, default: False
        Whether to use a low-memory mode for processing large slides.
    threshold : float, default: 0.5
        The threshold to binarize the probability map for segmentation.
    ignore_index : int or None, default: 0
        The index to ignore during segmentation (e.g., background).
    buffer_px : int, default: 2
        The buffer in pixels to apply when merging polygons.
    chunk_size : int, default: 512
        The size of chunks to process at a time when merging probability maps.
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model (e.g., "cpu" or "cuda"). If None, automatically selected.
    pbar : bool, default: True
        Whether to show the progress bar.
    key_added : str, default: "anatomical_structures"
        The key for the added instance shapes in the WSIData object.

    Returns
    -------
    None
        The segmentation results are added to the WSIData object under the specified key.

    """
    runner = SemanticSegmentationRunner(
        wsi=wsi,
        model=model,
        tile_key=tile_key,
        transform=transform,
        mode=mode,
        sigma_scale=sigma_scale,
        low_memory=low_memory,
        threshold=threshold,
        ignore_index=ignore_index,
        buffer_px=buffer_px,
        chunk_size=chunk_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        pbar=pbar,
        class_names=class_names,
    )
    shapes = runner.run()
    # Add the segmentation results to the WSIData
    add_shapes(wsi, key=key_added, shapes=shapes.explode().reset_index(drop=True))


def _initialize_merging_prob_masks(out: np.ndarray, height: int, width: int):
    """Create the masks for merging based on the shape of probability mask."""
    # TODO: decide the channel dimension
    dims = out.shape
    if len(dims) == 3:
        # If the input is a 3D tensor, it is likely in the format [C, H, W]
        C = out.shape[0]
    elif len(dims) == 4:
        # If the input is a 4D tensor, it is likely in the format [B, C, H, W]
        C = out.shape[1]
    else:
        raise ValueError(f"Unsupported shape for probability mask: {out.shape}")

    return np.zeros((C, height, width), dtype=np.float32)


def create_importance_map(
    patch_size: (int, int),
    sigma_scale: float = 0.125,
    mode: Literal["constant", "gaussian"] = "gaussian",
):
    if mode == "constant":
        return torch.ones(patch_size)
    elif mode == "gaussian":
        sigmas = torch.tensor(patch_size) * sigma_scale

        importance_map = None
        for i, size in enumerate(patch_size):
            x = torch.arange(
                start=-(size - 1) / 2.0,
                end=(size - 1) / 2.0 + 1,
                dtype=torch.float,
            )
            x = torch.exp(x**2 / (-2 * sigmas[i] ** 2))  # 1D gaussian
            if i == 0:
                importance_map = x
            else:
                importance_map = importance_map.unsqueeze(-1) * x.unsqueeze(0)
        return importance_map
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Supported modes are 'constant' and 'gaussian'."
        )


class Runner(ABC):
    @abstractmethod
    def run(self) -> gpd.GeoDataFrame:
        """
        Run the segmentation.
        This method should be implemented by subclasses.
        """
        pass

    def __call__(self):
        """
        Call the run method.
        This method is provided for convenience and can be overridden by subclasses.
        """
        return self.run()

    @staticmethod
    def tiler(array, tile_size=512):
        C, H, W = array.shape

        if H <= tile_size and W <= tile_size:
            # Image is smaller than the tile size — yield once
            yield array, (0, H, 0, W)
        else:
            for i in range(0, max(H, 1), tile_size):
                for j in range(0, max(W, 1), tile_size):
                    i_end = min(i + tile_size, H)
                    j_end = min(j + tile_size, W)
                    tile = array[:, i:i_end, j:j_end]
                    yield tile, (i, i_end, j, j_end)


class TileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wsi: WSIData,
        tiles: gpd.GeoDataFrame,
        tile_spec: TileSpec,
        transform: Callable = None,
    ):
        self.tiles_xy = tiles.bounds[["minx", "miny"]].to_numpy()
        self.tile_spec = tile_spec
        self.reader = wsi.reader
        self.reader.detach_reader()
        self.transform = transform

    def __len__(self):
        return len(self.tiles_xy)

    def __getitem__(self, idx):
        x, y = self.tiles_xy[idx]
        # Read the tile image from the WSI
        img = self.reader.get_region(
            x,
            y,
            width=self.tile_spec.ops_width,
            height=self.tile_spec.ops_height,
            level=self.tile_spec.ops_level,
        )
        img = self.reader.resize_img(
            img, dsize=[self.tile_spec.width, self.tile_spec.height]
        )
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "x": x,
            "y": y,
        }


class SemanticSegmentationRunner(Runner):
    def __init__(
        self,
        wsi: WSIData,
        model: SegmentationModel,
        tile_key: str = Key.tiles,
        transform: Callable = None,
        mode: Literal["constant", "gaussian"] = "gaussian",
        sigma_scale: float = 0.125,
        low_memory: bool = False,
        threshold: float = 0.5,
        ignore_index: int | None = 0,
        class_names: List[str] | Mapping[int, str] | None = None,
        buffer_px: int = 2,
        chunk_size: int = 512,
        batch_size: int = 4,
        num_workers: int = 0,
        device: str | None = None,
        pbar: bool = True,
    ):
        self.wsi = wsi
        self.model = model
        self.tile_key = tile_key
        self.transform = transform or model.get_transform()
        self.mode = mode
        self.low_memory = low_memory
        self.sigma_scale = sigma_scale
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.buffer_px = buffer_px
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or get_torch_device()
        self.model.to(self.device)
        self.class_names = class_names
        self.pbar = pbar

        self.tile_spec = wsi.tile_spec(tile_key)
        tissue_key = self.tile_spec.tissue_name
        self.has_tissue = tissue_key in wsi
        # No tissue associated with the tile key
        # Uses the whole image as a single tissue

        if not self.has_tissue:
            wsi_bounds = wsi.properties.bounds
            tissues = gpd.GeoDataFrame(
                {
                    "tissue_id": [0],
                    "geometry": [box(0, 0, wsi_bounds[2], wsi_bounds[3])],
                }
            )
        else:
            tissues = wsi[tissue_key]
        self.tissues = tissues
        self.downsample = self.tile_spec.base_downsample
        self._supported_output = self.model.supported_output()

    @cached_property
    def importance_map(self):
        """
        Create the importance map based on the patch size and sigma scale.
        The importance map is used to weight the model output.
        """
        patch_size = (self.tile_spec.height, self.tile_spec.width)
        return create_importance_map(
            patch_size=patch_size,
            sigma_scale=self.sigma_scale,
            mode=self.mode,
        )

    def run(self) -> gpd.GeoDataFrame:
        # For each tissue, we will run the segmentation
        results = []
        with default_pbar(disable=not self.pbar) as progress_bar:
            for _, row in self.tissues.iterrows():
                tid = row["tissue_id"]
                tissue = row["geometry"]
                minx, miny, maxx, maxy = tissue.bounds
                height, width = maxy - miny, maxx - minx
                # map the tissue bounds to the tile mpp
                height, width = (
                    int(height / self.downsample),
                    int(width / self.downsample),
                )

                # Create masks for 1) the output probabilities, 2) the count map
                prob_mask = None
                count_mask = torch.zeros((height, width), dtype=torch.float)

                # Get tiles within the tissue bounds
                if self.has_tissue:
                    current_tiles = self.wsi[self.tile_key][
                        self.wsi[self.tile_key]["tissue_id"] == tid
                    ]
                else:
                    current_tiles = self.wsi[self.tile_key]

                if len(current_tiles) > 0:
                    ds = TileDataset(
                        wsi=self.wsi,
                        tiles=current_tiles,
                        tile_spec=self.tile_spec,
                        transform=self.transform,
                    )
                    dl = DataLoader(
                        ds, batch_size=self.batch_size, num_workers=self.num_workers
                    )

                    task = progress_bar.add_task(
                        f"Processing tissue {tid}", total=len(ds)
                    )

                    for chunk in dl:
                        images = chunk["image"]
                        xs, ys = np.asarray(chunk["x"]), np.asarray(chunk["y"])
                        if self.device is not None:
                            images = images.to(self.device)
                        # TODO: output may not be tensor
                        output = self.model.segment(images)

                        probability_map = output["probability_map"]

                        if isinstance(probability_map, torch.Tensor):
                            # Update the out tensor with the importance map
                            probability_map = probability_map * self.importance_map.to(
                                probability_map.device
                            )
                            # Get output back to cpu
                            probability_map = probability_map.detach().cpu().numpy()
                        elif isinstance(probability_map, np.ndarray):
                            probability_map *= self.importance_map.numpy()
                        else:
                            raise TypeError(
                                f"Probability map type {type(probability_map)} is not supported"
                            )

                        # Calculate the position of the tile in the tissue bounds
                        for i in range(len(xs)):
                            # Update the probability mask
                            if prob_mask is None:
                                # Initialize the probability mask with the shape of the output
                                prob_mask = _initialize_merging_prob_masks(
                                    probability_map[i], height, width
                                )
                            pos_x = int((xs[i] - minx) / self.downsample)
                            pos_y = int((ys[i] - miny) / self.downsample)
                            slice_y = slice(
                                pos_y,
                                np.clip(
                                    pos_y + self.tile_spec.height,
                                    a_max=height,
                                    a_min=None,
                                ),
                            )
                            slice_x = slice(
                                pos_x,
                                np.clip(
                                    pos_x + self.tile_spec.width,
                                    a_max=width,
                                    a_min=None,
                                ),
                            )
                            # Clip out if it exceeds the mask boundaries
                            out_clipped = probability_map[i][
                                :,
                                : slice_y.stop - slice_y.start,
                                : slice_x.stop - slice_x.start,
                            ]
                            prob_mask[:, slice_y, slice_x] += out_clipped
                            # Update the count mask
                            count_mask[slice_y, slice_x] += 1
                        progress_bar.update(task, advance=len(images))
                        progress_bar.refresh()
                # Normalize the probability mask by the count mask
                prob_mask /= count_mask.unsqueeze(0).clamp(min=1e-6)
                prob_mask[prob_mask < 1e-3] = 0
                # Chunk the probability mask into PATCHES to avoid large memory allocation
                np_mask = prob_mask.detach().cpu().numpy()
                seg_objects = []

                for tile, (i, i_end, j, j_end) in self.tiler(
                    np_mask, tile_size=self.chunk_size
                ):
                    if tile.sum() > 0:
                        # TODO: There could be other output types,
                        #       e.g. multiclass/multilabel mask
                        m = ProbabilityMap(tile, class_names=self.class_names)
                        df = m.to_polygons(
                            threshold=self.threshold, ignore_index=self.ignore_index
                        )
                        if len(df) > 0:
                            df["geometry"] = df["geometry"].translate(
                                xoff=j,
                                yoff=i,
                            )
                            seg_objects.append(df)
                # Concatenate the results for this tissue
                if len(seg_objects) == 0:
                    continue
                seg_results = pd.concat(seg_objects, ignore_index=True).reset_index(
                    drop=True
                )
                # Move the polygons to the global coordinate
                seg_results["geometry"] = (
                    seg_results["geometry"]
                    .scale(xfact=self.downsample, yfact=self.downsample, origin=(0, 0))
                    .translate(xoff=minx, yoff=miny)
                    .buffer(0)
                )
                for class_id, class_group in seg_results.groupby("class"):
                    merged = merge_connected_polygons(
                        class_group,
                        prob_col="prob",
                        buffer_px=self.buffer_px,
                    )
                    # Filter out polygons that are outside the tissue bounds
                    merged = merged[merged.intersects(tissue)]
                    merged["class"] = class_id
                    results.append(merged)
            progress_bar.refresh()
        # Concatenate all results into a single GeoDataFrame
        if len(results) == 0:
            return gpd.GeoDataFrame(columns=["geometry"])
        return gpd.GeoDataFrame(pd.concat(results, ignore_index=True)).reset_index(
            drop=True
        )


class CellSegmentationRunner(Runner):
    def __init__(
        self,
        wsi: WSIData,
        model: SegmentationModel,
        tile_key: str = Key.tiles,
        transform: Callable = None,
        size_filter: bool = True,
        nucleus_size: (int, int) = (20, 1000),
        batch_size: int = 4,
        num_workers: int = 0,
        device: str | None = None,
        class_names: List[str] | Mapping[int, str] | None = None,
        pbar: bool = True,
    ):
        self.wsi = wsi
        self.model = model
        self.tile_key = tile_key
        self.transform = transform or model.get_transform()
        self.size_filter = size_filter
        self.nucleus_size = nucleus_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or get_torch_device()
        self.model.to(self.device)
        self.class_names = class_names
        self.pbar = pbar

        self.tile_spec = wsi.tile_spec(tile_key)
        self.downsample = self.tile_spec.base_downsample
        self._supported_output = self.model.supported_output()
        if "instance_map" not in self._supported_output:
            raise ValueError("The model does not support instance segmentation.")

    def run(self) -> gpd.GeoDataFrame:
        with default_pbar(disable=not self.pbar) as progress_bar:
            tile_dataset = self.wsi.ds.tile_images(
                tile_key=self.tile_key, transform=self.transform
            )

            tile_loader = DataLoader(
                tile_dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )

            results = []
            # is_classification = "class_map" in self._supported_output

            task = progress_bar.add_task("Processing tiles", total=len(tile_dataset))

            for chunk in tile_loader:
                images = chunk["image"]
                xs, ys = np.asarray(chunk["x"]), np.asarray(chunk["y"])
                if self.device is not None:
                    images = images.to(self.device)
                output = self.model.segment(images)

                instance_map = output["instance_map"]
                class_map = output.get("class_map", None)

                # Get output and covert to numpy
                if isinstance(instance_map, torch.Tensor):
                    instance_map = instance_map.detach().cpu().to(torch.int).numpy()
                if class_map is not None:
                    if isinstance(class_map, torch.Tensor):
                        class_map = class_map.detach().cpu().numpy()
                for i in range(len(xs)):
                    pos_x = xs[i]
                    pos_y = ys[i]
                    out = instance_map[i]
                    if class_map is not None:
                        prob_map = class_map[i]
                    else:
                        prob_map = None

                    # Convert the output to polygons
                    m = InstanceMap(
                        out,
                        prob_map=prob_map,
                        class_names=self.class_names,
                    )
                    df = m.to_polygons(detect_holes=False)
                    if len(df) > 0:
                        # Remove the polygons that are on the edge of the tile
                        tile_box = (
                            box(0, 0, self.tile_spec.width, self.tile_spec.height)
                            .buffer(-2)
                            .boundary
                        )
                        prepare(tile_box)
                        sel = df["geometry"].apply(
                            lambda geom: not tile_box.intersects(geom)
                        )
                        df = df[sel]
                        # Move the polygons to the global coordinate
                        df["geometry"] = (
                            df["geometry"]
                            .scale(
                                xfact=self.downsample,
                                yfact=self.downsample,
                                origin=(0, 0),
                            )
                            .translate(xoff=pos_x, yoff=pos_y)
                            .buffer(0)
                        )
                        if self.size_filter:
                            df = df[df["geometry"].area.between(*self.nucleus_size)]
                        results.append(df)
                progress_bar.update(task, advance=len(images))
            progress_bar.refresh()
        # Concatenate all results into a single GeoDataFrame
        cells = gpd.GeoDataFrame(pd.concat(results, ignore_index=True)).reset_index(
            drop=True
        )
        if len(cells) == 0:
            return gpd.GeoDataFrame(columns=["geometry"])
        if "prob" not in cells:
            cells["prob"] = 1
        # Drop the overlapping cells, preserving the largest one
        cells = nms(cells, "prob")
        # Remove cells that are not in the tissue
        tissue_key = self.tile_spec.tissue_name
        tissues = self.wsi[tissue_key]  # GeoDataFrame
        cells = cells[cells.intersects(tissues.unary_union)]

        return cells
