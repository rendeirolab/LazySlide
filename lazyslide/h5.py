from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from .utils import TileOps


class H5File:
    """A class that handle storage and reading of h5 file"""

    COORDS_KEY = "coords"
    MASKS_KEY = "masks"

    def __init__(self, h5_file):
        h5_file = Path(h5_file)
        mode = "r+" if h5_file.exists() else "w"
        self.file = h5py.File(h5_file, mode)
        self.coords = self.load_coords()
        self.tile_ops = self.load_tile_ops()
        self.masks = self.load_masks()
        self._rewrite = False
        self._rewrite_mask = True

    def __exit__(self, *args):
        """Close h5 handler when exit"""
        self.file.close()

    def set_coords(self, coords):
        # Delete the previous exist one
        self.coords = coords
        self._rewrite = True

    def load_coords(self):
        if self.COORDS_KEY in self.file:
            return self.file[self.COORDS_KEY][:]

    def get_coords(self):
        if self.coords is None:
            self.coords = self.load_coords()
        return self.coords

    def set_tile_ops(self, tile_ops: TileOps):
        self.tile_ops = tile_ops
        self._rewrite = True

    def load_tile_ops(self):
        if self.COORDS_KEY in self.file:
            h5_attrs = self.file[self.COORDS_KEY].attrs
            attrs = {}
            for key in h5_attrs.keys():
                attrs[key] = h5_attrs.get(key)
            return TileOps(**attrs)

    def get_tile_ops(self):
        if self.tile_ops is None:
            self.tile_ops = self.load_tile_ops()
        return self.tile_ops

    def set_mask(self, name, mask):
        self.masks[name] = mask
        self._rewrite_mask = True

    def load_masks(self):
        if self.MASKS_KEY in self.file:
            masks = {}
            masks_group = self.file[self.MASKS_KEY]
            for mask_name in masks_group.keys():
                masks[mask_name] = masks_group.get(mask_name)[:]
            return masks
        return {}

    def get_masks(self):
        if self.masks is None:
            self.masks = self.load_masks()
        return self.masks

    def save(self):
        if self._rewrite:
            # Delete the previous exist coords
            if self.COORDS_KEY in self.file:
                del self.file[self.COORDS_KEY]

            shape = self.coords.shape
            ds = self.file.create_dataset(self.COORDS_KEY, shape,
                                          dtype=np.uint16,
                                          chunks=True)
            ds[:] = self.coords
            attrs = ds.attrs
            for k, v in asdict(self.tile_ops).items():
                attrs[k] = v

        if self._rewrite_mask:
            # Delete the previous exist masks
            if self.MASKS_KEY in self.file:
                del self.file[self.MASKS_KEY]

            masks_group = self.file.create_group("masks")
            for mask_name, mask_array in self.masks.items():
                ds = masks_group.create_dataset(mask_name, shape=mask_array.shape,
                                                dtype=np.uint16, chunks=True)
                ds[:] = mask_array
