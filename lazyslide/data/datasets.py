from typing import List

import numpy as np
import pandas as pd

from lazyslide import WSI

try:
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    DataLoader = object


class TileImagesDataset(Dataset):
    def __init__(
        self,
        wsi: WSI,
        transform=None,
        target_transform=None,
        key: str = "tiles",
        target_key: str = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        # Check if the tile exists
        if key not in wsi.sdata.points:
            raise ValueError(f"Tile {key} not found.")
        # Do not assign wsi to self to avoid pickling
        tiles = wsi.sdata.points[key]
        get_keys = ["x", "y"]
        if target_key is not None:
            get_keys.append(target_key)
        if hasattr(tiles, "compute"):
            tiles = tiles[get_keys].compute()
        self.tiles = tiles[["x", "y"]].to_numpy()
        self.spec = wsi.get_tile_spec(key)

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

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        idx = self.ix_mapper[idx]
        x, y = self.tiles[idx]
        tile = self.reader.get_region(
            x, y, self.spec.width, self.spec.height, level=self.spec.level
        )
        if self.transform:
            tile = self.transform(tile)
        if self.targets is not None:
            tile_target = self.targets[idx]
            if self.target_transform:
                tile_target = self.target_transform(tile_target)
            return tile, tile_target
        return tile


class TileFeatureDataset(Dataset):
    pass


class GraphDataset(Dataset):
    pass


class WSIListDataset(Dataset):
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


class ImageLoader(DataLoader):
    pass


class DirectoryDataset(Dataset):
    """Load Images/Features/Graph from a directory."""

    pass
