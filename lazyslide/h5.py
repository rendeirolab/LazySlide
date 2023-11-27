from dataclasses import astuple
from pathlib import Path

import h5py

from .utils import TileOps


class H5File:
    """A class that handle storage and reading of h5 file"""

    COORDS_KEY = "coords"

    def __init__(self, h5_file):
        h5_file = Path(h5_file)
        mode = "r+" if h5_file.exists() else "w"
        self.file = h5py.File(h5_file, mode)
        self.coords = self.get_coords()
        self.tile_ops = self.get_tile_ops()
        self._rewrite = False

    def __exit__(self, *args):
        """Close h5 handler when exit"""
        self.file.close()

    def set_coords(self, coords):
        # Delete the previous exist one
        self.coords = coords
        self._rewrite = True

    def get_coords(self):
        if self.COORDS_KEY in self.file:
            return self.file[self.COORDS_KEY][:]

    def set_tile_ops(self, tile_ops: TileOps):
        self.tile_ops = tile_ops
        self._rewrite = True

    def get_tile_ops(self):
        if self.COORDS_KEY in self.file:
            h5_attrs = self.file[self.COORDS_KEY].attrs
            attrs = {}
            for key in h5_attrs.keys():
                attrs[key] = h5_attrs.get(key)
            return TileOps(**attrs)

    def save(self):
        if self._rewrite:
            # Delete the previous exist one
            if self.COORDS_KEY in self.file:
                del self.file[self.COORDS_KEY]

            shape = self.coords.shape
            ds = self.file.create_dataset(self.COORDS_KEY, shape, chunks=True)
            ds[:] = self.coords
            attrs = ds.attrs
            for k, v in astuple(self.tile_ops):
                attrs[k] = v

