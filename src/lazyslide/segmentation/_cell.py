from __future__ import annotations

import warnings

import torch
from lazyslide_models import MODEL_REGISTRY, SegmentationModelProtocol
from wsidata import WSIData
from wsidata.io import add_features, add_shapes

from .._const import Key
from .._utils import find_stack_level
from ._seg_runner import CellSegmentationRunner


def cells(
    wsi: WSIData,
    model: str | SegmentationModelProtocol = "instanseg",
    tile_key=Key.tiles,
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
    key_added="cells",
    **model_kwargs,
):
    """:term:`cell segmentation <Cell segmentation>` for the whole slide image.

    :term:`tile <Tiles>` should be prepared before segmentation, the tile size should be
    reasonable (with :term:`mpp` around 0.5) for the model to work properly

    Supported models:

    - instanseg :cite:p:`Goldsborough2024-oc`
    - cellpose :cite:p:`Stringer2021-cx`

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The :term:`WSIData` object to work on.
    model : str or SegmentationModelProtocol, default: "instanseg"
        The cell segmentation model.
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
    key_added : str, default: "cells"
        The key for the added cell shapes.

    Returns
    -------
    None
        The cell shapes are added to the :bdg-danger:`shapes` slot
        of the WSIData object.

    """

    if isinstance(model, SegmentationModelProtocol):
        model_instance = model
    else:
        model = MODEL_REGISTRY.get(model)
        if model is None:
            raise ValueError(f"Unknown model: {model}")
        model_instance = model(**model_kwargs)
    # Run tile check
    tile_spec = wsi.tile_spec(tile_key)
    if tile_spec is None:
        raise ValueError(
            f"Tiles for {tile_key} not found. Did you forget to run zs.pp.tile_tissues ?"
        )
    if model_instance.check_input_tile(tile_spec):
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
        )
        result = runner.run()
        if extract_features:
            cells_gdf, features = result
        else:
            cells_gdf = result
        # Add cells to the WSIData
        cells_gdf = cells_gdf.explode().reset_index(drop=True)
        add_shapes(wsi, key=key_added, shapes=cells_gdf)
        if extract_features and features.size > 0:
            add_features(
                wsi, key=f"{key_added}_features", tile_key=key_added, features=features
            )


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
    key_added="cell_types",
):
    """:term:`Cell type segmentation` for the :term:`whole slide image`.

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
    key_added : str, default: "cell_types"
        The key for the added cell type shapes.

    Returns
    -------
    None
        The cell type shapes are added to the :bdg-danger:`shapes` slot
        of the WSIData object.

    """

    tile_spec = wsi.tile_spec(tile_key)
    if tile_spec is None:
        raise ValueError(
            f"Tiles for {tile_key} not found. Did you forget to run zs.pp.tile_tissues ?"
        )

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

    if isinstance(model, SegmentationModelProtocol):
        model_instance = model
    else:
        model_kwargs = dict(magnification=magnification)
        model = MODEL_REGISTRY.get(model)
        if model is None:
            raise ValueError(f"Unknown model: {model}")

        model_instance = model(**model_kwargs)

    if model_instance.check_input_tile(tile_spec):
        runner = CellSegmentationRunner(
            wsi,
            model_instance,
            tile_key,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            size_filter=size_filter,
            nucleus_size=nucleus_size,
            amp=amp,
            autocast_dtype=autocast_dtype,
            pbar=pbar,
            extract_features=extract_features,
        )
        result = runner.run()
        if extract_features:
            cells_gdf, features = result
        else:
            cells_gdf = result
        # Exclude background
        bg_mask = cells_gdf["class"] != "Background"
        cells_gdf = cells_gdf[bg_mask]
        if extract_features and features.size > 0:
            features = features[bg_mask.values]
        cells_gdf = cells_gdf.explode().reset_index(drop=True)
        add_shapes(wsi, key=key_added, shapes=cells_gdf)
        if extract_features and features.size > 0:
            add_features(
                wsi, key=f"{key_added}_features", tile_key=key_added, features=features
            )
