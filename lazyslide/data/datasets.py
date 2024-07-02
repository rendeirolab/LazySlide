from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from lazyslide import WSI

import lazy_loader as lazy

torch = lazy.load("torch")


class TileImagesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wsi: WSI,
        key: str = "tiles",
        target_key: str = None,
        transform=None,
        color_norm=None,
        target_transform=None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        # Do not assign wsi to self to avoid pickling
        tiles = wsi.get_tiles_table(key)
        self.tiles = tiles[["x", "y"]].to_numpy()
        self.spec = wsi.get_tile_spec(key)
        self.color_norm = color_norm

        self.targets = None
        if target_key is not None:
            self.targets = tiles[target_key].to_numpy()
        self.transform = transform
        self.target_transform = target_transform

        ix = np.arange(len(self.tiles))
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(ix)
        self.ix_mapper = dict(enumerate(ix))
        # Send reader to the worker instead of wsi
        self.reader = wsi.reader
        self.reader.detach_reader()

    @cached_property
    def cn_func(self):
        return self.get_cn_func()

    def get_cn_func(self):
        if self.color_norm is not None:
            from lazyslide.cv.colornorm import ColorNormalizer

            cn = ColorNormalizer(method=self.color_norm)
            return lambda x: cn(x)
        else:
            return lambda x: x

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        idx = self.ix_mapper[idx]
        x, y = self.tiles[idx]
        tile = self.reader.get_region(
            x, y, self.spec.width, self.spec.height, level=self.spec.level
        )
        self.cn_func(tile)
        if self.transform:
            tile = self.transform(tile)
        if self.targets is not None:
            tile_target = self.targets[idx]
            if self.target_transform:
                tile_target = self.target_transform(tile_target)
            return tile, tile_target
        return tile


class TileImagesDiskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        key="tiles",
        transform=None,
        target_transform=None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.tile_dir = Path(root) / "tile_images" / key
        self.table = pd.read_csv(self.tile_dir / "tiles.csv")


class TileFeatureDataset(torch.utils.data.Dataset):
    pass


class GraphDataset(torch.utils.data.Dataset):
    pass


class WSIListDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wsi_list: List[WSI | str],
        targets: pd.DataFrame = None,
        transform=None,
        target_transform=None,
        wsi_init_fn=None,
    ):
        self.wsi_list = wsi_list
        self.transform = transform

        if targets is not None:
            self.targets = targets.to_numpy()


class DirectoryDataset(torch.utils.data.Dataset):
    """Load Images/Features/Graph from a directory."""

    pass
