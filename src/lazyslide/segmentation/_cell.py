from __future__ import annotations

import warnings

from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide.models import SegmentationModel
from lazyslide.models.segmentation import Instanseg, NuLite
from ._seg_runner import SegmentationRunner
from .._const import Key


def cells(
    wsi: WSIData,
    model: str | SegmentationModel = "instanseg",
    tile_key=Key.tiles,
    transform=None,
    batch_size=4,
    num_workers=0,
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
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    key_added : str, default: "cells"
        The key for the added cell shapes.

    """
    if model == "instanseg":
        model = Instanseg()
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

    runner = SegmentationRunner(
        wsi,
        model,
        tile_key,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    cells = runner.run()
    # Add cells to the WSIData
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
    wsi : WSIData
        The whole slide image data.
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

    model = NuLite()

    runner = SegmentationRunner(
        wsi,
        model,
        tile_key,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    cells = runner.run()
    # Add cells to the WSIData
    add_shapes(wsi, key=key_added, shapes=cells)
