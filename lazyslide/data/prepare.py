from __future__ import annotations

from typing import List, Literal

from lazyslide.wsi import WSI

DATA_TYPES = Literal[
    "tile_images",
    "tissue_images",
    "cell_images",
    "tile_features",
    "tissue_features",
    "cell_features",
    "graph",
]


# TODO: The feature should be saved using safetensors


def prepare_disk_dataset(
    wsi: WSI,
    ouput_dir: str,
    data_types: str | List[str] = "tiles",
    tile_key: str = "tiles",
    feature_key: str = None,
):
    """A utility function to prepare a dataset on disk.

    This is useful for fast data loading and processing during model training.

    """
    pass
