from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from h5py import Empty

from .utils import TileOps


class H5File:
    """A class that handles storage and reading of h5 file"""

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

    def set_coords(self, coords):
        # Chunk by row-major order
        self._save_dataset(self.COORDS_KEY, coords, chunks=True)

    def get_coords(self):
        return self._load_dataset(self.COORDS_KEY)

    def get_one_coord_by_index(self, index):
        return self._load_dataset_by_slice(self.COORDS_KEY, index)

    def _load_dataset(self, key: str, group=None):
        with h5py.File(self.file, "r") as h5:
            if group is not None:
                if group not in h5:
                    return None
                h5 = h5[group]
            if key in h5:
                ds = h5[key]
                return ds[:]
            else:
                return None

    def _load_dataset_by_slice(self, key: str, s: slice, group=None):
        with h5py.File(self.file, "r") as h5:
            if group is not None:
                if group not in h5:
                    return None
                h5 = h5[group]
            if key in h5:
                ds = h5[key]
                return ds[s]
            else:
                return None

    def _load_attrs(self, key: str, group=None):
        with h5py.File(self.file, "r") as h5:
            if group is not None:
                if group not in h5:
                    return None
                h5 = h5[group]
            if key in h5:
                ds = h5[key]
                h5_attrs = ds.attrs
                attrs = {}
                for key in h5_attrs.keys():
                    value = h5_attrs.get(key)
                    if isinstance(value, Empty):
                        value = None
                    attrs[key] = value
                return attrs
            else:
                return None

    def _save_dataset(self, key: str, data: np.ndarray, group=None, **kwargs):
        with h5py.File(self.file, "r+") as h5:
            if group is not None:
                h5 = h5.require_group(group)
            if key in h5:
                del h5[key]
            h5.create_dataset(
                key,
                data=data,
                dtype=np.uint32,
                **kwargs,
            )

    def _save_attr(self, key: str, attrs: dict, group=None):
        with h5py.File(self.file, "r+") as h5:
            if group is not None:
                h5 = h5.require_group(group)
            if key in h5:
                ds = h5[key]
                for k, v in attrs.items():
                    if v is None:
                        v = Empty(dtype="f")
                    ds.attrs[k] = v

    def _has_dataset(self, key: str, group=None):
        with h5py.File(self.file, "r") as h5:
            if group is not None:
                if group not in h5:
                    return False
                h5 = h5[group]
            return key in h5

    def set_tile_ops(self, tile_ops: TileOps):
        if self._has_dataset(self.COORDS_KEY):
            new_attrs = asdict(tile_ops)
            for k, v in new_attrs.items():
                if v is None:
                    new_attrs[k] = Empty(dtype="f")
            self._save_attr(self.COORDS_KEY, new_attrs)
        else:
            raise ValueError("Please set coords first")

    def get_tile_ops(self):
        attrs = self._load_attrs(self.COORDS_KEY)
        if attrs is None:
            return None
        return TileOps(**attrs)

    def set_mask(self, name: str, mask: np.ndarray, level: int):
        self._save_dataset(name, mask, group=self.MASKS_KEY, chunks=False)
        self._save_attr(name, {"level": level}, group=self.MASKS_KEY)

    def get_masks(self, name) -> (np.ndarray, int):
        if not self._has_dataset(name, group=self.MASKS_KEY):
            return None, None
        return self._load_dataset(name, group=self.MASKS_KEY), self._load_attrs(
            name, group=self.MASKS_KEY
        )["level"]

    def get_available_masks(self):
        with h5py.File(self.file, "r") as h5:
            if self.MASKS_KEY in h5:
                masks_group = h5[self.MASKS_KEY]
                return list(masks_group.keys())
            else:
                return []

    def set_contours_holes(self, contours, holes):
        for i, arr in enumerate(contours):
            self._save_dataset(f"{self.CONTOURS_KEY}_{i}", arr, group=self.CONTOURS_KEY)
        for i, arr in enumerate(holes):
            self._save_dataset(f"{self.HOLES_kEY}_{i}", arr, group=self.HOLES_kEY)
        self._save_attr(
            self.CONTOURS_KEY, {"length": len(contours)}, group=self.CONTOURS_KEY
        )
        self._save_attr(self.HOLES_kEY, {"length": len(holes)}, group=self.HOLES_kEY)

    def get_contours_holes(self):
        contours = []
        holes = []
        n_contours = self._load_attrs(self.CONTOURS_KEY, group=self.CONTOURS_KEY)
        n_holes = self._load_attrs(self.HOLES_kEY, group=self.HOLES_kEY)
        if n_contours is None or n_holes is None:
            return [], []
        else:
            n_contours = n_contours["length"]
            n_holes = n_holes["length"]
        for i in range(n_contours):
            contours.append(
                self._load_dataset(f"{self.CONTOURS_KEY}_{i}", group=self.CONTOURS_KEY)
            )
        for i in range(n_holes):
            holes.append(
                self._load_dataset(f"{self.HOLES_kEY}_{i}", group=self.HOLES_kEY)
            )

        return contours, holes

    @property
    def has_tiles(self):
        return self._has_dataset(self.COORDS_KEY)

    @property
    def has_masks(self):
        return self._has_dataset(self.MASKS_KEY)

    @property
    def has_contours_holes(self):
        return self._has_dataset(self.CONTOURS_KEY) and self._has_dataset(
            self.HOLES_kEY
        )
