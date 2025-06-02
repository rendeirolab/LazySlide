from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial, cached_property
from typing import Literal, Callable, Mapping

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely import box
from shapely.affinity import scale, translate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wsidata import WSIData, TileSpec
from wsidata.io import add_shapes

from lazyslide._const import Key
from lazyslide._utils import default_pbar, get_torch_device
from lazyslide.cv import merge_polygons, ProbabilityMap
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


def slide_window_inference(
    wsi: WSIData,
    model: SegmentationModel,
    tile_key: str = Key.tiles,
    mode: Literal["constant", "gaussian"] = "gaussian",
    sigma_scale: float = 0.125,
    device: str | None = None,
):
    spec = wsi.tile_spec(tile_key)
    tile_height, tile_width = spec.height, spec.width
    patch_size = (spec.height, spec.width)
    downsample = spec.base_downsample

    importance_map = create_importance_map(
        patch_size, sigma_scale=sigma_scale, mode=mode
    )

    transform = model.get_transform()
    device = get_torch_device() if device is None else device
    model.to(device)

    tissue_key = spec.tissue_name
    # Tissue may not be found in the WSIData, so we check if it exists
    tissues = wsi[tissue_key]
    tiles = wsi[tile_key]
    for _, row in tissues.iterrows():
        tid = row["tissue_id"]
        tissue = row["geometry"]
        minx, miny, maxx, maxy = tissue.bounds
        height, width = maxy - miny, maxx - minx
        # map the tissue bounds to the tile mpp
        height, width = int(height / downsample), int(width / downsample)
        # Create masks for 1) the output probabilities, 2) the importance map, 3) the count map
        prob_mask = None
        count_mask = torch.zeros((height, width), dtype=torch.float)

        # Get tiles within the tissue bounds
        current_tiles = tiles[tiles["tissue_id"] == tid]
        for _, tile in tqdm(current_tiles.iterrows(), total=len(current_tiles)):
            tile_bounds = tile["geometry"].bounds
            tile_x, tile_y = tile_bounds[0:2]
            img = wsi.read_region(
                tile_x,
                tile_y,
                width=spec.ops_width,
                height=spec.ops_height,
                level=spec.ops_level,
            )
            img = cv2.resize(
                img, (spec.width, spec.height), interpolation=cv2.INTER_AREA
            )

            # _, ax = plt.subplots(ncols=2)
            # ax[0].imshow(img)

            img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
            # Apply the model to the tile
            # TODO: Redesign the SegmentationModel API
            img = transform(img)

            out = model.segment(img.to(device))
            out = out.squeeze(0)
            if prob_mask is None:
                # Initialize the probability mask with the shape of the output
                prob_mask = _initialize_merging_prob_masks(out, height, width)
            # Update the out tensor with the importance map
            out = out * importance_map.to(out.device)
            # Get out back to cpu
            out = out.detach().cpu()
            # ax[1].imshow(out.permute(1, 2, 0).cpu().numpy())
            # plt.show()

            # Calculate the position of the tile in the tissue bounds
            pos_x = int((tile_x - minx) / downsample)
            pos_y = int((tile_y - miny) / downsample)
            # Update the probability mask
            # print(f"Updating mask at position ({pos_x}, {pos_y}) with shape {out.shape} {prob_mask.shape}")
            # print(f"Range: ({pos_y}:{pos_y + tile_height}, {pos_x}:{pos_x + tile_width})")
            slice_y = slice(
                pos_y, np.clip(pos_y + tile_height, a_max=height, a_min=None)
            )
            slice_x = slice(pos_x, np.clip(pos_x + tile_width, a_max=width, a_min=None))
            # Clip out if it exceeds the mask boundaries
            out_clipped = out[
                :, : slice_y.stop - slice_y.start, : slice_x.stop - slice_x.start
            ]
            prob_mask[:, slice_y, slice_x] += out_clipped
            # Update the count mask
            count_mask[slice_y, slice_x] += 1

        # Normalize the probability mask by the count mask
        prob_mask /= count_mask.unsqueeze(0).clamp(min=1e-6)
        return prob_mask
    return None


def _initialize_merging_prob_masks(out: torch.Tensor, height: int, width: int):
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

    return torch.zeros((C, height, width), dtype=torch.float)


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
    def run(self):
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
        buffer_px: int = 2,
        chunk_size: int = 512,
        batch_size: int = 4,
        num_workers: int = 0,
        device: str | None = None,
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

        self.tile_spec = wsi.tile_spec(tile_key)
        tissue_key = self.tile_spec.tissue_name
        # No tissue associated with the tile key
        # Uses the whole image as a single tissue

        if tissue_key not in wsi:
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

    def run(self):
        # For each tissue, we will run the segmentation
        results = []
        for _, row in self.tissues.iterrows():
            tid = row["tissue_id"]
            tissue = row["geometry"]
            minx, miny, maxx, maxy = tissue.bounds
            height, width = maxy - miny, maxx - minx
            # map the tissue bounds to the tile mpp
            height, width = int(height / self.downsample), int(width / self.downsample)

            # Create masks for 1) the output probabilities, 2) the importance map, 3) the count map
            prob_mask = None
            count_mask = torch.zeros((height, width), dtype=torch.float)

            # Get tiles within the tissue bounds
            current_tiles = self.wsi[self.tile_key][
                self.wsi[self.tile_key]["tissue_id"] == tid
            ]

            ds = TileDataset(
                wsi=self.wsi,
                tiles=current_tiles,
                tile_spec=self.tile_spec,
                transform=self.transform,
            )
            dl = DataLoader(
                ds, batch_size=self.batch_size, num_workers=self.num_workers
            )

            for chunk in tqdm(dl, desc=f"Processing tissue {tid}"):
                images = chunk["image"]
                xs, ys = np.asarray(chunk["x"]), np.asarray(chunk["y"])
                if self.device is not None:
                    images = images.to(self.device)
                output = self.model.segment(images)

                # Update the out tensor with the importance map
                output = output * self.importance_map.to(output.device)

                # Get output back to cpu
                output = output.detach().cpu()

                # Calculate the position of the tile in the tissue bounds
                for i in range(len(xs)):
                    pos_x = int((xs[i] - minx) / self.downsample)
                    pos_y = int((ys[i] - miny) / self.downsample)
                    # Update the probability mask
                    if prob_mask is None:
                        # Initialize the probability mask with the shape of the output
                        prob_mask = _initialize_merging_prob_masks(
                            output[i], height, width
                        )
                    slice_y = slice(
                        pos_y,
                        np.clip(
                            pos_y + self.tile_spec.height, a_max=height, a_min=None
                        ),
                    )
                    slice_x = slice(
                        pos_x,
                        np.clip(pos_x + self.tile_spec.width, a_max=width, a_min=None),
                    )
                    # Clip out if it exceeds the mask boundaries
                    out_clipped = output[i][
                        :,
                        : slice_y.stop - slice_y.start,
                        : slice_x.stop - slice_x.start,
                    ]
                    prob_mask[:, slice_y, slice_x] += out_clipped
                    # Update the count mask
                    count_mask[slice_y, slice_x] += 1
            # Normalize the probability mask by the count mask
            prob_mask /= count_mask.unsqueeze(0).clamp(min=1e-6)
            # Chunk the probability mask into PATCHES to avoid large memory allocation
            np_mask = prob_mask.detach().cpu().numpy()
            seg_objects = []
            for tile, (i, i_end, j, j_end) in self.tiler(
                np_mask, tile_size=self.chunk_size
            ):
                if tile.sum() > 0:
                    m = ProbabilityMap(tile)
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
            merged_results = merge_polygons(
                seg_results,
                class_col="class",
                prob_col="prob",
                buffer_px=self.buffer_px,
            )
            # Filter out polygons that are outside the tissue bounds
            merged_results = merged_results[merged_results.intersects(tissue)]
            results.append(merged_results)

        # Concatenate all results into a single GeoDataFrame
        return gpd.GeoDataFrame(pd.concat(results, ignore_index=True)).reset_index(
            drop=True
        )


class CellSegmentationRunner(Runner):
    pass
