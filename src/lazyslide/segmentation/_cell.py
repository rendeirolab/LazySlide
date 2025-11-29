from __future__ import annotations

import warnings

import torch
from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide.models import MODEL_REGISTRY, SegmentationModel

from .._const import Key
from .._utils import find_stack_level
from ._seg_runner import CellSegmentationRunner


def cells(
    wsi: WSIData,
    model: str | SegmentationModel = "instanseg",
    tile_key=Key.tiles,
    transform=None,
    batch_size=4,
    num_workers=0,
    device=None,
    amp: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
    size_filter=False,
    nucleus_size=(20, 1000),
    pbar=True,
    key_added="cells",
    **model_kwargs,
):
    """Cell segmentation for the whole slide image.

    Tiles should be prepared before segmentation, the tile size should be
    reasonable (with mpp around 0.5) for the model to work properly

    Supported models:

    - instanseg :cite:p:`Goldsborough2024-oc`
    - cellpose :cite:p:`Stringer2021-cx`

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : str | SegmentationModel, default: "instanseg"
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
    pbar : bool, default: True
        Whether to show a progress bar during segmentation.
    key_added : str, default: "cells"
        The key for the added cell shapes.

    """

    if isinstance(model, SegmentationModel):
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
        )
        cells = runner.run()
        # Add cells to the WSIData
        add_shapes(wsi, key=key_added, shapes=cells.explode().reset_index(drop=True))


def cell_types(
    wsi: WSIData,
    model: str | SegmentationModel = "nulite",
    tile_key=Key.tiles,
    magnification: str | None = None,
    transform=None,
    batch_size=4,
    num_workers=0,
    device=None,
    amp: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
    size_filter=False,
    nucleus_size=(20, 1000),
    pbar=True,
    key_added="cell_types",
):
    """Cell type segmentation for the whole slide image.

    Tiles should be prepared before segmentation, the tile size should be
    reasonable (with mpp around 0.5) for the model to work properly

    Supported models:
        - nulite: :cite:p:`Tommasino2024-tg`
        - histoplus: :cite:p:`Adjadj2025-hn`

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : str | SegmentationModel, default: "nulite"
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
    pbar : bool, default: True
        Whether to show a progress bar during segmentation.
    key_added : str, default: "cell_types"
        The key for the added cell type shapes.

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
    assert magnification in {"20x", "40x"}, (
        f"Unsupported magnification: {magnification}, use '20x' or '40x'"
    )

    if isinstance(model, SegmentationModel):
        model_instance = model
    else:
        model_kwargs = dict(magnification=magnification)
        if model == "histoplus":
            model_kwargs["tile_size"] = tile_spec.height
        model = MODEL_REGISTRY.get(model)
        if model is None:
            raise ValueError(f"Unknown model: {model}")

        model_instance = model(**model_kwargs)

    CLASS_MAPPING = model.get_classes()
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
            class_names=CLASS_MAPPING,
        )
        cells = runner.run()
        # Add cells to the WSIData
        # Exclude background
        cells = cells[cells["class"] != "Background"]
        cells = cells.explode().reset_index(drop=True)
        add_shapes(wsi, key=key_added, shapes=cells)
