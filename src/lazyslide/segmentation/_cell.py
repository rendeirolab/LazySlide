from __future__ import annotations

import warnings

from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide.models import SegmentationModel

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
    if model == "instanseg":
        from lazyslide.models.segmentation import Instanseg

        model = Instanseg(**model_kwargs)
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
    elif model == "cellpose":
        from lazyslide.models.segmentation import Cellpose

        model = Cellpose(**model_kwargs)
    else:
        if not isinstance(model, SegmentationModel):
            raise ValueError(f"Unknown model: {model}")

    runner = CellSegmentationRunner(
        wsi,
        model,
        tile_key,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        size_filter=size_filter,
        nucleus_size=nucleus_size,
        device=device,
        pbar=pbar,
    )
    cells = runner.run()
    # Add cells to the WSIData
    add_shapes(wsi, key=key_added, shapes=cells.explode().reset_index(drop=True))


def cell_types(
    wsi: WSIData,
    model: str | SegmentationModel = "nulite",
    tile_key=Key.tiles,
    transform=None,
    batch_size=4,
    num_workers=0,
    device=None,
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

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : str | SegmentationModel, default: "nulite"
        The cell type segmentation model.
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
    key_added : str, default: "cell_types"
        The key for the added cell type shapes.

    """

    if model == "nulite":
        from lazyslide.models.segmentation import NuLite

        model = NuLite()
        # Run tile check
        tile_spec = wsi.tile_spec(tile_key)
        check_mpp = tile_spec.mpp == 0.5 or tile_spec.mpp == 0.25
        # check_size = tile_spec.height == 102 and tile_spec.width == 512
        if not check_mpp:
            warnings.warn(
                f"To optimize the performance of NuLite model, "
                f"the tiles should be created at the mpp=0.5 or 0.25. "
                f"Current tile size is {tile_spec.width}x{tile_spec.height} with {tile_spec.mpp} mpp."
            )
        CLASS_MAPPING = {
            0: "Background",
            1: "Neoplastic",
            2: "Inflammatory",
            3: "Connective",
            4: "Dead",
            5: "Epithelial",
        }
    else:
        raise ValueError(
            "Currently only 'nulite' model is supported for cell type segmentation."
        )

    runner = CellSegmentationRunner(
        wsi,
        model,
        tile_key,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        size_filter=size_filter,
        nucleus_size=nucleus_size,
        pbar=pbar,
        class_names=CLASS_MAPPING,
    )
    cells = runner.run()
    # Add cells to the WSIData
    # Exclude background
    cells = cells[cells["class"] != "Background"]
    cells = cells.explode().reset_index(drop=True)
    add_shapes(wsi, key=key_added, shapes=cells)


def nulite(
    wsi: WSIData,
    tile_key="tiles",
    transform=None,
    batch_size=4,
    num_workers=0,
    device=None,
    key_added="cell_types",
):
    """Cell type segmentation for the whole slide image.

    Tiles should be prepared before segmentation.

    Recommended tile setting:
    - **nulite**: 512x512, mpp=0.5

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
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
    key_added : str, default: "cell_types"
        The key for the added cell type shapes.

    """
    warnings.warn(
        "Deprecated since v0.7.0 and will be removed in v0.8: Use `cell_types` instead.",
        DeprecationWarning,
        stacklevel=find_stack_level(),
    )

    return cell_types(
        wsi,
        model="nulite",
        tile_key=tile_key,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
