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
        # Create new file if not exist
        if not h5_file.exists():
            with h5py.File(h5_file, "w"):
                pass
        self.file = h5_file
        self.coords = None
        self.tile_ops = None
        self.masks = None
        self.load()

        self._rewrite = False
        self._rewrite_mask = True

    def set_coords(self, coords):
        # Delete the previous exist one
        self.coords = coords
        self._rewrite = True

    def load(self):
        with h5py.File(self.file, "r+") as h5:
            if self.COORDS_KEY in h5:
                self.coords = h5[self.COORDS_KEY][:]
                h5_attrs = h5[self.COORDS_KEY].attrs
                attrs = {}
                for key in h5_attrs.keys():
                    value = h5_attrs.get(key)
                    if key == "mask_name" and value == 0:
                        value = None
                    attrs[key] = value
                self.tile_ops = TileOps(**attrs)

            if self.MASKS_KEY in h5:
                masks = {}
                masks_group = h5[self.MASKS_KEY]
                for mask_name in masks_group.keys():
                    masks[mask_name] = masks_group.get(mask_name)[:]
                self.masks = masks
            self.masks = {}

    def get_coords(self):
        return self.coords

    def set_tile_ops(self, tile_ops: TileOps):
        self.tile_ops = tile_ops
        self._rewrite = True

    def get_tile_ops(self):
        return self.tile_ops

    def set_mask(self, name, mask):
        self.masks[name] = mask
        self._rewrite_mask = True

    def get_masks(self):
        return self.masks

    def save(self):
        with h5py.File(self.file, "r+") as h5:
            if self._rewrite:
                # Delete the previous exist coords
                if self.COORDS_KEY in h5:
                    del h5[self.COORDS_KEY]

                shape = self.coords.shape
                ds = h5.create_dataset(self.COORDS_KEY, shape,
                                       dtype=np.uint16,
                                       chunks=True)
                ds[:] = self.coords
                attrs = ds.attrs
                for k, v in asdict(self.tile_ops).items():
                    if k == "mask_name" and v is None:
                        v = 0
                    attrs[k] = v

            if self._rewrite_mask:
                # Delete the previous exist masks
                if self.MASKS_KEY in h5:
                    del h5[self.MASKS_KEY]

                masks_group = h5.create_group("masks")
                for mask_name, mask_array in self.masks.items():
                    ds = masks_group.create_dataset(mask_name, shape=mask_array.shape,
                                                    dtype=np.uint16, chunks=True)
                    ds[:] = mask_array
