from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from wsidata import WSIData
from wsidata.io import add_shapes

from .._const import Key
from .._utils import find_stack_level
from ._seg_runner import CellSegmentationRunner

if TYPE_CHECKING:
    import torch
    from lazyslide_models import SegmentationModelProtocol


def _features_to_anndata(features: np.ndarray, cell_ids: np.ndarray) -> AnnData:
    """Build AnnData with `cell_id` as the join column on `.obs`.

    The AnnData has one row per unique cell. `obs["cell_id"]` (int64) is the
    join key against the `cell_id` column on the shapes GeoDataFrame.
    """
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    obs = pd.DataFrame({"cell_id": cell_ids}, index=cell_ids.astype(str))
    return AnnData(X=features, obs=obs)


def _add_cell_features(wsi: WSIData, key: str, tile_key: str, feat_adata: AnnData):
    """Write a per-cell features AnnData into ``wsi.tables[key]``.

    Bypasses ``wsidata.io.add_features`` because that helper hard-codes
    ``tile_id = arange(len(features))`` and would clobber our ``cell_id`` /
    crash when broadcasting ``library_id`` against a non-default obs index.
    """
    from spatialdata.models import TableModel

    n = feat_adata.n_obs
    feat_adata.obs["tile_id"] = feat_adata.obs["cell_id"].astype(np.int64).values
    feat_adata.obs["library_id"] = pd.Categorical([tile_key] * n, categories=[tile_key])
    table = TableModel.parse(
        feat_adata,
        region=tile_key,
        region_key="library_id",
        instance_key="tile_id",
    )
    wsi.tables[key] = table


def cells(
    wsi: WSIData,
    model: str | SegmentationModelProtocol = "instanseg",
    tile_key=Key.tiles,
    magnification: str | None = None,
    transform=None,
    batch_size=4,
    num_workers=0,
    device=None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    size_filter=False,
    nucleus_size=(20, 1000),
    pbar=True,
    extract_features: bool = False,
    low_memory: bool = False,
    key_added="cells",
    **model_kwargs,
):
    """:term:`cell segmentation <Cell segmentation>` for the whole slide image.

    :term:`tile <Tiles>` should be prepared before segmentation, the tile size should be
    reasonable (with :term:`mpp` around 0.5) for the model to work properly

    Supported models:

    - instanseg :cite:p:`Goldsborough2024-oc`
    - cellpose :cite:p:`Stringer2021-cx`
    - nulite :cite:p:`Tommasino2024-tg`
    - histoplus :cite:p:`Adjadj2025-hn`

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The :term:`WSIData` object to work on.
    model : str or SegmentationModelProtocol, default: "instanseg"
        The cell segmentation model.
    tile_key : str, default: "tiles"
        The key of the tile table.
    magnification : str, default: None
        The magnification of the model. Used by cell type segmentation models.
    transform : callable, default: None
        The transformation for the input tiles.
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    amp : bool, optional
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, optional
        The dtype for automatic mixed precision.
    size_filter : bool, default: False
        Whether to filter cells by nucleus size.
    nucleus_size : tuple of (int, int), default: (20, 1000)
        The (min, max) nucleus size range in pixels for filtering.
        Only used when ``size_filter=True``.
    pbar : bool, default: True
        Whether to show a progress bar during segmentation.
    extract_features : bool, default: False
        Whether to extract per-cell feature vectors from the model's
        ``patch_token_map``.  Only available for ViT-based segmentation
        models (e.g. NuLite, HistoPLUS).  If the model does not produce
        a ``patch_token_map``, a warning is emitted and features are skipped.
    low_memory : bool, default: False
        When ``extract_features=True``, write intermediate feature chunks to
        on-disk mmap files and only materialize features that survive filtering
        and NMS.
    key_added : str, default: "cells"
        The key for the added cell shapes.

    Returns
    -------
    None
        The cell shapes are added to the :bdg-danger:`shapes` slot
        of the WSIData object.

    """

    tile_spec = wsi.tile_spec(tile_key)
    if tile_spec is None:
        raise ValueError(
            f"Tiles for {tile_key} not found. Did you forget to run zs.pp.tile_tissues ?"
        )

    if not isinstance(model, str):
        model_name = None
    else:
        model_name = model

    if model_name in {"nulite", "histoplus"} or magnification is not None:
        magnification = _infer_magnification(tile_spec, tile_key, magnification)
        model_kwargs.setdefault("magnification", magnification)

    from lazyslide_models import MODEL_REGISTRY, SegmentationModelProtocol

    if isinstance(model, SegmentationModelProtocol):
        model_instance = model
    else:
        model_cls = MODEL_REGISTRY.get(model)
        if model_cls is None:
            raise ValueError(f"Unknown model: {model}")
        model_instance = model_cls(**model_kwargs)

    runner = CellSegmentationRunner(
        wsi,
        model_instance,
        tile_key,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        size_filter=size_filter,
        nucleus_size=nucleus_size,
        device=device,
        amp=amp,
        autocast_dtype=autocast_dtype,
        pbar=pbar,
        extract_features=extract_features,
        low_memory=low_memory,
    )
    result = runner.run()
    if extract_features:
        cells_gdf, features, feature_cell_ids = result
    else:
        cells_gdf = result

    if "class" in cells_gdf.columns:
        bg_mask = cells_gdf["class"] != "Background"
        cells_gdf = cells_gdf[bg_mask]
        if extract_features and features.size > 0:
            kept_ids = set(cells_gdf["cell_id"].tolist())
            keep_feat = np.array(
                [cid in kept_ids for cid in feature_cell_ids], dtype=bool
            )
            features = features[keep_feat]
            feature_cell_ids = np.asarray(feature_cell_ids)[keep_feat]

    # One row per cell. A single cell may be a MultiPolygon (e.g. ``buffer(0)``
    # split a pinched mask into disjoint parts). Keep it as ONE row so the
    # shape count stays 1:1 with the per-cell feature rows. Exploding here
    # would inflate the shape count past the feature count.
    cells_gdf = cells_gdf.reset_index(drop=True)
    if len(cells_gdf) == 0:
        return
    add_shapes(wsi, key=key_added, shapes=cells_gdf)
    if extract_features and features.size > 0:
        feat_adata = _features_to_anndata(features, feature_cell_ids)
        _add_cell_features(
            wsi,
            key=f"{key_added}_features",
            tile_key=key_added,
            feat_adata=feat_adata,
        )


def _infer_magnification(tile_spec, tile_key: str, magnification: str | None) -> str:
    if magnification is None:
        if tile_spec.mpp is None:
            warnings.warn(
                f"Mpp not found for {tile_key}. Will use model trained at magnification 20x.",
                stacklevel=find_stack_level(),
            )
            magnification = "20x"
        elif 0.6 >= tile_spec.mpp >= 0.4:  # Heuristic
            magnification = "20x"
        elif 0.3 >= tile_spec.mpp >= 0.1:  # Heuristic
            magnification = "40x"
        else:
            raise ValueError(
                f"Requested tiles are generated at {tile_spec.mpp} mpp, "
                f"only magnifications 20x (mpp=0.5) and 40x (mpp=0.25) are supported."
                f"You can either generate new tiles or pass `magnification='20x'/'40x'` to select "
                f"which model to use."
            )
    assert magnification in {
        "20x",
        "40x",
    }, f"Unsupported magnification: {magnification}, use '20x' or '40x'"
    return magnification


def cell_types(
    wsi: WSIData,
    model: str | SegmentationModelProtocol = "nulite",
    tile_key=Key.tiles,
    magnification: str | None = None,
    transform=None,
    batch_size=4,
    num_workers=0,
    device=None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    size_filter=False,
    nucleus_size=(20, 1000),
    pbar=True,
    extract_features: bool = False,
    low_memory: bool = False,
    key_added="cell_types",
    **model_kwargs,
):
    """:term:`Cell type segmentation` for the :term:`whole slide image`.

    .. deprecated::

        Use :func:`cells` instead.

    :term:`tile <Tiles>` should be prepared before segmentation, the tile size should be
    reasonable (with :term:`mpp` around 0.5) for the model to work properly

    Supported models:
        - nulite: :cite:p:`Tommasino2024-tg`
        - histoplus: :cite:p:`Adjadj2025-hn`

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : str or SegmentationModelProtocol, default: "nulite"
        The cell type segmentation model.
    tile_key : str, default: "tiles"
        The key of the tile table.
    magnification : str, default: None
        The magnification of the model.
    transform : callable, default: None
        The transformation for the input tiles.
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    amp : bool, optional
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, optional
        The dtype for automatic mixed precision.
    size_filter : bool, default: False
        Whether to filter cells by nucleus size.
    nucleus_size : tuple of (int, int), default: (20, 1000)
        The (min, max) nucleus size range in pixels for filtering.
        Only used when ``size_filter=True``.
    pbar : bool, default: True
        Whether to show a progress bar during segmentation.
    extract_features : bool, default: False
        Whether to extract per-cell feature vectors from the model's
        ``patch_token_map``.
    low_memory : bool, default: False
        When ``extract_features=True``, write intermediate feature chunks to
        on-disk mmap files and only materialize features that survive filtering
        and NMS.
    key_added : str, default: "cell_types"
        The key for the added cell type shapes.

    Returns
    -------
    None
        The cell type shapes are added to the :bdg-danger:`shapes` slot
        of the WSIData object.

    """

    warnings.warn(
        "`zs.seg.cell_types` is deprecated; use `zs.seg.cells` instead.",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    return cells(
        wsi,
        model=model,
        tile_key=tile_key,
        magnification=magnification,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        amp=amp,
        autocast_dtype=autocast_dtype,
        size_filter=size_filter,
        nucleus_size=nucleus_size,
        pbar=pbar,
        extract_features=extract_features,
        low_memory=low_memory,
        key_added=key_added,
        **model_kwargs,
    )
