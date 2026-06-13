from __future__ import annotations

import tempfile
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import cached_property
from typing import TYPE_CHECKING, Callable, List, Literal, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box, prepare
from wsidata import TileSpec, WSIData
from wsidata.io import add_shapes

from lazyslide import _api
from lazyslide._const import Key
from lazyslide._utils import default_pbar, find_stack_level, get_torch_device
from lazyslide.cv import (
    InstanceMap,
    ProbabilityMap,
    nms,
)

if TYPE_CHECKING:
    import torch
    from lazyslide_models import SegmentationModelProtocol


def _pool_cell_features(
    instance_map: np.ndarray,
    patch_token_map: np.ndarray,
    instance_ids,
) -> dict[int, np.ndarray]:
    """Mean-pool patch tokens for each instance in ``instance_ids``.

    For every instance, tokens are averaged over the patches whose
    nearest-neighbour downsampled location falls inside the instance — the same
    pooling as a per-instance ``(instance_map == id)`` mask, computed for all
    instances at once.

    Parameters
    ----------
    instance_map : np.ndarray, shape ``[H, W]``
        Integer instance ID map for the tile.
    patch_token_map : np.ndarray, shape ``[D, PH, PW]``
        Patch token feature map from a ViT segmentation model.
    instance_ids : sequence of int
        The instance IDs to extract features for.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``instance_id -> mean-pooled feature vector [D]``.
    """
    D, PH, PW = patch_token_map.shape
    H, W = instance_map.shape
    ids = np.asarray(list(instance_ids))
    if ids.size == 0:
        return {}

    # Nearest-neighbour downsample of the instance labels to patch resolution,
    # done once and reused for every instance.
    row_idx = np.round(np.linspace(0, H - 1, PH)).astype(int)
    col_idx = np.round(np.linspace(0, W - 1, PW)).astype(int)
    inst_patch = instance_map[np.ix_(row_idx, col_idx)].reshape(-1)  # [P]
    tokens = patch_token_map.reshape(D, -1)  # [D, P]

    # Membership of each patch to each requested instance: [K, P].
    membership = inst_patch[None, :] == ids[:, None]
    counts = membership.sum(axis=1)  # [K]
    # Grouped sum of tokens per instance via one matmul: [K, P] @ [P, D] -> [K, D]
    sums = membership.astype(tokens.dtype) @ tokens.T

    features: dict[int, np.ndarray] = {}
    hit = counts > 0
    hit_idx = np.flatnonzero(hit)
    if hit_idx.size:
        means = sums[hit_idx] / counts[hit_idx, None].astype(tokens.dtype)
        for k, idx in enumerate(hit_idx):
            features[int(ids[idx])] = means[k]

    # Fallback: instances too small to land on any patch grid point use the
    # nearest patch token to their full-resolution centroid. Compute every such
    # centroid in a SINGLE pass over the full-res map (one np.isin + grouped
    # bincount) rather than one ``np.where`` scan per missed instance.
    miss_idx = np.flatnonzero(~hit)
    if miss_idx.size:
        miss_ids = ids[miss_idx]
        flat = instance_map.reshape(-1)
        sel = np.isin(flat, miss_ids)
        Km = miss_ids.size
        cnt = np.zeros(Km, dtype=np.int64)
        sum_y = np.zeros(Km, dtype=np.float64)
        sum_x = np.zeros(Km, dtype=np.float64)
        if sel.any():
            lin = np.flatnonzero(sel)
            order = np.argsort(miss_ids)
            slot = order[np.searchsorted(miss_ids[order], flat[lin])]
            cnt = np.bincount(slot, minlength=Km)
            sum_y = np.bincount(slot, weights=lin // W, minlength=Km)
            sum_x = np.bincount(slot, weights=lin % W, minlength=Km)
        for j, idx in enumerate(miss_idx):
            inst = int(ids[idx])
            if cnt[j] == 0:
                features[inst] = np.zeros(D, dtype=patch_token_map.dtype)
                continue
            py = min(int((sum_y[j] / cnt[j]) * PH / H), PH - 1)
            px = min(int((sum_x[j] / cnt[j]) * PW / W), PW - 1)
            features[inst] = patch_token_map[:, py, px]
    return features


def _nms_by_tissue(
    cells: gpd.GeoDataFrame,
    tissues: gpd.GeoDataFrame,
    prob_col: str,
) -> gpd.GeoDataFrame:
    """Run NMS independently for each tissue row.

    Cells are assigned to the first tissue piece they intersect so a cell that
    touches multiple tissue geometries is not duplicated in the output.
    """
    if len(cells) == 0:
        return cells
    if len(tissues) == 0:
        return cells.iloc[[]].reset_index(drop=True)

    chunks = []
    assigned = np.zeros(len(cells), dtype=bool)
    for tissue_geom in tissues.geometry:
        if tissue_geom is None or tissue_geom.is_empty:
            continue
        mask = (~assigned) & cells.geometry.intersects(tissue_geom).to_numpy()
        if not mask.any():
            continue
        assigned[mask] = True
        chunk = nms(cells.iloc[np.flatnonzero(mask)], prob_col)
        if len(chunk) > 0:
            chunks.append(chunk)

    if len(chunks) == 0:
        return cells.iloc[[]].reset_index(drop=True)
    return gpd.GeoDataFrame(
        pd.concat(chunks, ignore_index=True),
        crs=cells.crs,
    ).reset_index(drop=True)


class _CellFeatureStore:
    def __init__(self, low_memory: bool = False):
        self.low_memory = low_memory
        self._tmpdir = tempfile.TemporaryDirectory() if low_memory else None
        self._chunks = []
        self._id_chunks: list[np.ndarray] = []
        self._dtype = None
        self._dim = None

    def append(self, features: np.ndarray, cell_ids: np.ndarray):
        if features.size == 0:
            return
        features = np.asarray(features)
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        if self._dtype is None:
            self._dtype = features.dtype
            self._dim = features.shape[1]
        self._id_chunks.append(cell_ids.copy())
        if self.low_memory:
            path = f"{self._tmpdir.name}/cell_features_{len(self._chunks)}.npy"
            chunk = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=features.dtype,
                shape=features.shape,
            )
            chunk[:] = features
            chunk.flush()
            self._chunks.append(path)
        else:
            self._chunks.append(features)

    def __len__(self) -> int:
        return len(self._chunks)

    def select(
        self,
        surviving_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_ids = np.concatenate(self._id_chunks).astype(np.int64, copy=False)
        surviving_ids = np.asarray(surviving_ids, dtype=np.int64)
        order = np.argsort(feature_ids)
        sorted_ids = feature_ids[order]
        loc = np.searchsorted(sorted_ids, surviving_ids)
        valid = loc < sorted_ids.size
        valid[valid] &= sorted_ids[loc[valid]] == surviving_ids[valid]
        loc = loc[valid]
        selected_ids = surviving_ids[valid]
        selected_pos = order[loc]
        if self.low_memory:
            features = self._select_low_memory(selected_pos)
        else:
            all_features = np.concatenate(self._chunks, axis=0)
            features = all_features[selected_pos]
        return features, selected_ids, valid

    def _select_low_memory(self, selected_pos: np.ndarray) -> np.ndarray:
        features = np.empty(
            (len(selected_pos), self._dim),
            dtype=self._dtype,
        )
        offsets = np.cumsum([0, *[len(ids) for ids in self._id_chunks]])
        for chunk_ix, path in enumerate(self._chunks):
            start, end = offsets[chunk_ix], offsets[chunk_ix + 1]
            mask = (selected_pos >= start) & (selected_pos < end)
            if not mask.any():
                continue
            chunk = np.load(path, mmap_mode="r")
            features[mask] = chunk[selected_pos[mask] - start]
        return features

    def close(self):
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


def semantic(
    wsi: WSIData,
    model: SegmentationModelProtocol,
    tile_key=Key.tiles,
    class_names: List[str] | Mapping[int, str] | None = None,
    transform=None,
    mode: Literal["constant", "gaussian"] = "constant",
    sigma_scale: float = 0.125,
    low_memory: bool = False,
    threshold: float = 0.5,
    ignore_index: int | None = 0,
    buffer_px: int = 2,
    chunk_size: int = 512,
    batch_size=4,
    num_workers=0,
    device=None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    pbar: bool = None,
    key_added="anatomical_structures",
):
    """
    :term:`Semantic segmentation` for the :term:`whole slide image <WSI>`.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : SegmentationModelProtocol
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
        The buffer in pixels to apply when merging :term:`polygons <polygon>`.
    chunk_size : int, default: 512
        The size of chunks to process at a time when merging probability maps.
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model (e.g., "cpu" or "cuda"). If None, automatically selected.
    amp : bool, optional
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, optional
        The dtype for automatic mixed precision.
    pbar : bool, optional
        Whether to show the progress bar.
    key_added : str, default: "anatomical_structures"
        The key for the added :term:`instance` shapes in the WSIData object.

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
        amp=amp,
        autocast_dtype=autocast_dtype,
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
    import torch

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


class TileDataset:
    """A map-style dataset over the tiles of a WSI.

    Deliberately a plain class rather than a ``torch.utils.data.Dataset``
    subclass. It uses no torch itself, and PyTorch's ``DataLoader`` accepts any
    object implementing ``__len__`` and ``__getitem__`` as a map-style dataset,
    so keeping it torch-free means importing this module does not import torch.

    Defining it at module level (instead of inside a closure) is what keeps it
    picklable: ``DataLoader(num_workers>0)`` pickles the dataset under the spawn
    start method (the default on macOS/Windows), which requires the class to be
    importable by its qualified name.
    """

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
        model: SegmentationModelProtocol,
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
        amp: bool = None,
        autocast_dtype: torch.dtype = None,
        pbar: bool = None,
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
        self.device = _api.default_value("device", device)
        self.model.to(self.device)
        self.amp = _api.default_value("amp", amp)
        self.autocast_dtype = _api.default_value("autocast_dtype", autocast_dtype)
        self.class_names = class_names
        self.pbar = _api.default_value("pbar", pbar)

        self.tile_spec = wsi.tile_spec(tile_key)
        tissue_key = self.tile_spec.tissue_name
        self.has_tissue = tissue_key in wsi
        # No tissue associated with the tile key
        # Uses the whole image as a single tissue

        if not self.has_tissue:
            wsi_bounds = wsi.properties.bounds
            bx, by, bw, bh = wsi_bounds
            tissues = gpd.GeoDataFrame(
                {
                    "tissue_id": [0],
                    "geometry": [box(bx, by, bx + bw, by + bh)],
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

    def run(self) -> gpd.GeoDataFrame:
        import torch
        from torch.utils.data import DataLoader

        # For each tissue, we will run the segmentation
        results = []
        with default_pbar(disable=not self.pbar) as progress_bar:
            amp_ctx = (
                torch.autocast(self.device, self.autocast_dtype)
                if self.amp
                else nullcontext()
            )
            with amp_ctx, torch.inference_mode():
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
                    # keep types consistent (numpy) to avoid subtle torch/numpy mixing issues
                    count_mask = np.zeros((height, width), dtype=np.float32)

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

                            probability_map = output.probability_map
                            if probability_map is None:
                                raise ValueError(
                                    "Semantic segmentation requires probability_map "
                                    "but the model returned None. "
                                    "This model may only support instance segmentation."
                                )

                            if isinstance(probability_map, torch.Tensor):
                                # Update the out tensor with the importance map
                                probability_map = (
                                    probability_map
                                    * self.importance_map.to(probability_map.device)
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
                    # Skip if no tiles were processed
                    if prob_mask is None:
                        continue
                    # Normalize the probability mask by the count mask
                    prob_mask /= np.clip(count_mask, 1e-6, None)[None, ...]
                    prob_mask[prob_mask < 1e-3] = 0
                    # Chunk the probability mask into PATCHES to avoid large memory allocation
                    np_mask = prob_mask
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
                        .scale(
                            xfact=self.downsample, yfact=self.downsample, origin=(0, 0)
                        )
                        .translate(xoff=minx, yoff=miny)
                        .buffer(0)
                    )
                    for class_id, class_group in seg_results.groupby("class"):
                        # Robust per-class dissolve to remove chunk seams:
                        # buffer -> union -> unbuffer, with tolerance in base-pixel units
                        tol = max(1.0, float(self.buffer_px) * float(self.downsample))
                        buffered = class_group.geometry.buffer(tol)
                        united = buffered.union_all()
                        cleaned = gpd.GeoDataFrame(geometry=[united.buffer(-tol)])
                        # Explode multi-geometries back to rows
                        cleaned = cleaned.explode(index_parts=False).reset_index(
                            drop=True
                        )
                        # Filter out polygons that are outside the tissue bounds
                        cleaned = cleaned[cleaned.intersects(tissue)]
                        cleaned["class"] = class_id
                        if "prob" in class_group:
                            # Use the max class confidence as representative for merged parts
                            cleaned["prob"] = float(class_group["prob"].max())
                        results.append(cleaned)
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
        model: SegmentationModelProtocol,
        tile_key: str = Key.tiles,
        transform: Callable = None,
        size_filter: bool = True,
        nucleus_size: (int, int) = (20, 1000),
        batch_size: int = 4,
        num_workers: int = 0,
        device: str | None = None,
        amp: bool = False,
        autocast_dtype: torch.dtype = None,
        class_names: List[str] | Mapping[int, str] | None = None,
        pbar: bool = True,
        extract_features: bool = False,
        low_memory: bool = False,
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
        self.amp = amp
        self.autocast_dtype = autocast_dtype
        self.class_names = class_names
        self.pbar = pbar
        self.extract_features = extract_features
        self.low_memory = low_memory

        self.tile_spec = wsi.tile_spec(tile_key)
        self.downsample = self.tile_spec.base_downsample

    def run(
        self,
    ) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
        import torch
        from torch.utils.data import DataLoader

        # Feature chunks track the same globally unique cell_id assigned to
        # polygon rows. In low-memory mode, chunks are backed by on-disk mmap.
        feature_store = _CellFeatureStore(
            low_memory=self.extract_features and self.low_memory
        )
        global_cell_idx = 0

        autocast_dtype = (
            self.autocast_dtype if self.autocast_dtype is not None else torch.float16
        )
        with default_pbar(disable=not self.pbar) as progress_bar:
            amp_ctx = (
                torch.autocast(self.device, autocast_dtype)
                if self.amp
                else nullcontext()
            )
            with amp_ctx, torch.inference_mode():
                tile_dataset = self.wsi.ds.tile_images(
                    tile_key=self.tile_key, transform=self.transform
                )

                tile_loader = DataLoader(
                    tile_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                )

                results = []
                _warned_no_tokens = False

                # Inner boundary of a tile. Cells touching it are clipped by the
                # tile edge and dropped (the same cell is captured whole in a
                # neighbouring/overlapping tile). Constant across tiles -> build once
                # and prepare it for fast repeated intersection tests.
                tile_box = (
                    box(0, 0, self.tile_spec.width, self.tile_spec.height)
                    .buffer(-2)
                    .boundary
                )
                prepare(tile_box)
                tissue_key = self.tile_spec.tissue_name
                tissues = self.wsi[tissue_key]
                tissue = tissues.union_all()
                prepare(tissue)

                task = progress_bar.add_task(
                    "Processing tiles", total=len(tile_dataset)
                )

                for chunk in tile_loader:
                    images = chunk["image"]
                    xs, ys = np.asarray(chunk["x"]), np.asarray(chunk["y"])
                    if self.device is not None:
                        images = images.to(self.device)
                    output = self.model.segment(images)

                    instance_map = output.instance_map
                    if instance_map is None:
                        raise ValueError(
                            "Cell segmentation requires instance_map "
                            "but the model returned None. "
                            "This model may only support semantic segmentation."
                        )
                    probability_map = output.probability_map
                    patch_token_map = output.patch_token_map

                    # Resolve class_names from output if not provided
                    if self.class_names is None and output.classes is not None:
                        self.class_names = {
                            i: name for i, name in enumerate(output.classes)
                        }

                    # Get output and convert to numpy
                    if isinstance(instance_map, torch.Tensor):
                        instance_map = instance_map.detach().cpu().to(torch.int).numpy()
                    if probability_map is not None:
                        if isinstance(probability_map, torch.Tensor):
                            probability_map = probability_map.detach().cpu().numpy()
                    if patch_token_map is not None:
                        if isinstance(patch_token_map, torch.Tensor):
                            patch_token_map = (
                                patch_token_map.detach().cpu().float().numpy()
                            )

                    has_tokens = self.extract_features and patch_token_map is not None
                    if (
                        self.extract_features
                        and patch_token_map is None
                        and not _warned_no_tokens
                    ):
                        warnings.warn(
                            "extract_features=True but model does not return "
                            "patch_token_map. Feature extraction will be skipped.",
                            stacklevel=find_stack_level(),
                        )
                        _warned_no_tokens = True

                    for i in range(len(xs)):
                        pos_x = xs[i]
                        pos_y = ys[i]
                        out = instance_map[i]
                        if probability_map is not None:
                            prob_map = probability_map[i]
                        else:
                            prob_map = None

                        # Convert the instance map to one polygon per instance.
                        # ``df`` carries an ``instance_id`` column linking each row
                        # back to its source instance in ``out``.
                        m = InstanceMap(
                            out,
                            prob_map=prob_map,
                            class_names=self.class_names,
                        )
                        df = m.to_polygons(detect_holes=False)
                        if len(df) == 0:
                            continue

                        # Drop cells touching the tile edge (clipped instances).
                        df = df[~df["geometry"].intersects(tile_box)]
                        if len(df) == 0:
                            continue

                        # Move the polygons to the global coordinate
                        df = df.copy()
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
                            if len(df) == 0:
                                continue
                        df = df[df["geometry"].intersects(tissue)]
                        if len(df) == 0:
                            continue
                        if "class" in df.columns:
                            df = df[df["class"] != "Background"]
                            if len(df) == 0:
                                continue

                        # Assign a globally-unique, stable cell_id to every cell.
                        n = len(df)
                        cell_ids = np.arange(
                            global_cell_idx, global_cell_idx + n, dtype=np.int64
                        )
                        global_cell_idx += n
                        df["cell_id"] = cell_ids

                        # Bind per-cell features by instance_id — exact, no centroid
                        # guessing. Tokens are pooled over each surviving instance's
                        # full mask, then looked up by the row's instance_id. Every
                        # row's instance_id produced this polygon, so a feature always
                        # exists and no cell is silently dropped.
                        if has_tokens:
                            token_map_i = patch_token_map[i]  # [D, PH, PW]
                            inst_arr = df["instance_id"].to_numpy()
                            cell_features = _pool_cell_features(
                                out, token_map_i, np.unique(inst_arr)
                            )
                            feature_store.append(
                                np.stack(
                                    [cell_features[int(inst)] for inst in inst_arr]
                                ),
                                cell_ids,
                            )

                        results.append(df.drop(columns="instance_id", errors="ignore"))
                    progress_bar.update(task, advance=len(images))
            progress_bar.refresh()
        # If no results
        empty = gpd.GeoDataFrame(columns=["geometry", "cell_id"])
        if len(results) == 0:
            feature_store.close()
            if self.extract_features:
                return empty, np.empty((0, 0)), np.empty((0,), dtype=np.int64)
            return empty
        # Concatenate all results into a single GeoDataFrame
        cells = gpd.GeoDataFrame(pd.concat(results, ignore_index=True)).reset_index(
            drop=True
        )
        # If all results are empty dataframe
        if len(cells) == 0:
            feature_store.close()
            if self.extract_features:
                return empty, np.empty((0, 0)), np.empty((0,), dtype=np.int64)
            return empty
        if "prob" not in cells:
            cells["prob"] = 1
        # Drop overlapping cells independently within each tissue piece.
        cells = _nms_by_tissue(cells, tissues, "prob")

        if self.extract_features:
            if len(feature_store) == 0:
                feature_store.close()
                # extract_features=True but model never returned patch_token_map.
                return (
                    cells,
                    np.empty((len(cells), 0)),
                    cells["cell_id"].to_numpy(dtype=np.int64),
                )
            surviving_ids = cells["cell_id"].to_numpy(dtype=np.int64)
            features, surviving_ids, valid = feature_store.select(surviving_ids)
            feature_store.close()
            if not valid.all():
                # Mixed-output models can produce some cells without token maps.
                cells = cells[valid].reset_index(drop=True)
            return cells, features, surviving_ids

        feature_store.close()
        return cells
