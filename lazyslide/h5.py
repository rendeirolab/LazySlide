from dataclasses import asdict
from pathlib import Path

import h5py
from h5py import Empty

from .utils import TileOps


class H5File:
    """A class that handle storage and reading of h5 file"""

    COORDS_KEY = "coords"
    MASKS_KEY = "masks"
    CONTOURS_KEY = "contours"
    HOLES_kEY = "holes"

    def __init__(self, h5_file):
        h5_file = Path(h5_file)
        # Create new file if not exist
        if not h5_file.exists():
            with h5py.File(h5_file, "w"):
                pass
        self.file = h5_file
        self.coords = None
        self.tile_ops = None
        self.masks = {}
        self.masks_level = {}
        self.contours = []
        self.holes = []
        self.load()

        self._rewrite = False
        self._rewrite_mask = False
        self._rewrite_contours = False

    def set_coords(self, coords):
        # Delete the previous exist one
        self.coords = coords
        self._rewrite = True

    def load(self):
        with h5py.File(self.file, "r") as h5:
            if self.COORDS_KEY in h5:
                ds = h5[self.COORDS_KEY]
                self.coords = ds[:]
                h5_attrs = ds.attrs
                attrs = {}
                for key in h5_attrs.keys():
                    value = h5_attrs.get(key)
                    if isinstance(value, Empty):
                        value = None
                    attrs[key] = value
                self.tile_ops = TileOps(**attrs)

            if self.MASKS_KEY in h5:
                masks = {}
                masks_level = {}
                masks_group = h5[self.MASKS_KEY]
                for mask_name in masks_group.keys():
                    ds = masks_group.get(mask_name)
                    masks[mask_name] = ds[:]
                    masks_level[mask_name] = ds.attrs['level']
                self.masks = masks
                self.masks_level = masks_level

            if self.CONTOURS_KEY in h5:
                contours = []
                holes = []

                contours_group = h5[self.CONTOURS_KEY]
                c_keys = contours_group.keys()
                for n in range(len(c_keys)):
                    ds = contours_group.get(f"{self.CONTOURS_KEY}_{n}")
                    contours.append(ds[:])

                holes_group = h5[self.HOLES_kEY]
                h_keys = holes_group.keys()
                for n in range(len(h_keys)):
                    ds = holes_group.get(f"{self.HOLES_kEY}_{n}")
                    holes.append(ds[:])
                self.contours = contours
                self.holes = holes

    def get_coords(self):
        return self.coords

    def set_tile_ops(self, tile_ops: TileOps):
        self.tile_ops = tile_ops
        self._rewrite = True

    def get_tile_ops(self):
        return self.tile_ops

    def set_mask(self, name, mask, level):
        self.masks[name] = mask
        self.masks_level[name] = level
        self._rewrite_mask = True

    def get_masks(self):
        return self.masks, self.masks_level

    def set_contours_holes(self, contours, holes):
        self.contours = contours
        self.holes = holes
        self._rewrite_contours = True

    def get_contours_holes(self):
        return self.contours, self.holes

    def save(self):
        with h5py.File(self.file, "r+") as h5:
            if self._rewrite:
                # Delete the previous exist coords
                if self.COORDS_KEY in h5:
                    del h5[self.COORDS_KEY]

                ds = h5.create_dataset(self.COORDS_KEY, data=self.coords,
                                       chunks=True)
                attrs = ds.attrs
                for k, v in asdict(self.tile_ops).items():
                    if v is None:
                        v = Empty(dtype="f")
                    attrs[k] = v

            if self._rewrite_mask:
                # Delete the previous exist masks
                if self.MASKS_KEY in h5:
                    del h5[self.MASKS_KEY]

                masks_group = h5.create_group(self.MASKS_KEY)
                for mask_name, mask_array in self.masks.items():
                    ds = masks_group.create_dataset(mask_name, data=mask_array, chunks=True)
                    attrs = ds.attrs
                    attrs['level'] = self.masks_level[mask_name]

            if self._rewrite_contours:
                if self.CONTOURS_KEY in h5:
                    del h5[self.CONTOURS_KEY]

                if self.HOLES_kEY in h5:
                    del h5[self.HOLES_kEY]

                contours_group = h5.create_group(self.CONTOURS_KEY)
                for i, arr in enumerate(self.contours):
                    dataset_name = f"{self.CONTOURS_KEY}_{i}"
                    contours_group.create_dataset(dataset_name, data=arr)

                holes_group = h5.create_group(self.HOLES_kEY)
                for i, arr in enumerate(self.holes):
                    dataset_name = f"{self.HOLES_kEY}_{i}"
                    holes_group.create_dataset(dataset_name, data=arr)
