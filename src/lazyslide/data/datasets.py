from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import Dataset

from wsi_data import WSIData


class TileImagesDataset(Dataset):
    def __init__(
        self,
        wsi: WSIData,
        key: str = "tiles",
        target_key: str = None,
        transform=None,
        color_norm=None,
        target_transform=None,
    ):
        # Do not assign wsi to self to avoid pickling
        tiles = wsi.sdata[key]
        self.tiles = tiles[["x", "y"]].to_numpy()
        self.spec = wsi.tile_spec(key)
        self.color_norm = color_norm

        self.targets = None
        if target_key is not None:
            self.targets = tiles[target_key].to_numpy()
        self.transform = transform
        self.target_transform = target_transform

        # Send reader to the worker instead of wsi
        self.reader = wsi.reader
        self.reader.detach_reader()

    @cached_property
    def cn_func(self):
        return self.get_cn_func()

    def get_cn_func(self):
        if self.color_norm is not None:
            from lazyslide_cv.colornorm import ColorNormalizer

            cn = ColorNormalizer(method=self.color_norm)
            return lambda x: cn(x)
        else:
            return lambda x: x

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        x, y = self.tiles[idx]
        tile = self.reader.get_region(
            x, y, self.spec.width, self.spec.height, level=self.spec.level
        )
        tile = self.cn_func(tile)
        if self.transform:
            tile = self.transform(tile)
        if self.targets is not None:
            tile_target = self.targets[idx]
            if self.target_transform:
                tile_target = self.target_transform(tile_target)
            return tile, tile_target
        return tile


class TileImagesDiskDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        key="tiles",
        transform=None,
        target_transform=None,
    ):
        self.tile_dir = Path(root) / "tile_images" / key
        self.table = pd.read_csv(self.tile_dir / "tiles.csv")


class TileFeatureDataset(Dataset):
    def __init__(
        self,
        wsi: WSIData,
        feature_key: str,
        target_key: str = None,
        target_transform=None,
    ):
        tables = wsi.get_features(feature_key)
        self.X = tables.X
        self.tables = tables.obs
        self.targets = None
        if target_key in self.tables:
            self.targets = self.tables[target_key].to_numpy()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.targets is not None:
            y = self.targets[idx]
            if self.target_transform is not None:
                y = self.target_transform(y)
            return x, y
        return x


class GraphDataset(Dataset):
    pass


class WSIListDataset(Dataset):
    def __init__(
        self,
        wsi_list: List[WSIData | str],
        targets: pd.DataFrame = None,
        transform=None,
        target_transform=None,
        wsi_init_fn=None,
    ):
        self.wsi_list = wsi_list
        self.transform = transform

        if targets is not None:
            self.targets = targets.to_numpy()


class DirectoryDataset(Dataset):
    """Load Images/Features/Graph from a directory."""

    pass