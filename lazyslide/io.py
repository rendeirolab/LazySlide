from __future__ import annotations

__all__ = ["H5ZSFile"]

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tables

from .utils import TileOps


class IOGroup:
    VERSION = "1.0"
    VENDOR = "lazyslide"

    def __init__(self, file: str | Path):
        file = Path(file)
        self.file = file
        # Create new file if not exist
        if not file.exists():
            with tables.open_file(file, "w"):  # type: ignore
                pass
            self._set_lazyslide_specs()
        else:
            self._check_lazyslide_specs()

    def _set_lazyslide_specs(self):
        with tables.open_file(self.file, "a"):  # type: ignore
            key = "lazyslide_specs"
            self._save_attributes(key, dict(version=self.VERSION, vendor=self.VENDOR))

    def _check_lazyslide_specs(self):
        with tables.open_file(self.file, "r"):  # type: ignore
            key = "lazyslide_specs"
            specs = self._read_attributes(key)
            if specs is None:
                RuntimeWarning(f"{self.file} is not a lazyslide file.")
            if specs["version"] != self.VERSION:
                raise RuntimeError(
                    f"lazyslide version mismatch: {specs['version']} != {self.VERSION}"
                )

    def _save_dataset(self, key: str, data: np.ndarray, group=None):
        with tables.open_file(self.file, "a") as h5:  # type: ignore
            if group is not None:
                if group not in h5.root:
                    slot = h5.create_group("/", group, createparents=True)
                else:
                    slot = h5.get_node(f"/{group}")
            else:
                slot = h5.root
            if key in slot:
                h5.remove_node(slot, key, recursive=True)
            h5.create_array(slot, key, data)

    def _save_attributes(self, key: str, attrs: dict):
        with tables.open_file(self.file, "a") as h5:  # type: ignore
            if key not in h5.root:
                group = h5.create_group("/", key)
            else:
                group = h5.root[key]
            for k, v in attrs.items():
                setattr(group._v_attrs, k, v)

    def _save_dataframe(self, key: str, data: pd.DataFrame):
        data.to_hdf(self.file, key=key)

    def _read_dataset(self, key: str, group=None):
        with tables.open_file(self.file, "r") as h5:  # type: ignore
            if group is not None:
                if group not in h5.root:
                    return None
                if key not in h5.get_node(f"/{group}"):
                    return None
                slot = h5.get_node(f"/{group}", key)
            else:
                if key not in h5.root:
                    return None
                slot = h5.get_node("/", key)
            return slot[:]

    def _read_dataframe(self, key: str):
        # Handle when key is missing
        if not self._has_dataset(key):
            return None
        return pd.read_hdf(self.file, key=key)

    def _read_attributes(self, key: str):
        with tables.open_file(self.file, "r") as h5:  # type: ignore
            if key in h5.root:
                attrs_node = h5.get_node("/", key)
                attrs = {}
                for name in attrs_node._v_attrs._f_list():
                    attrs[name] = attrs_node._v_attrs[name]
                return attrs
            else:
                return None

    def _delete_group(self, group=None):
        with tables.open_file(self.file, "a") as h5:  # type: ignore
            if group in h5.root:
                h5.remove_node("/", group, recursive=True)

    def _delete_dataset(self, key: str, group=None):
        with tables.open_file(self.file, "a") as h5:  # type: ignore
            if group is not None:
                if group not in h5.root:
                    return
                h5.remove_node(f"/{group}", key, recursive=True)
            else:
                if key in h5.root:
                    h5.remove_node("/", key, recursive=True)

    def _has_dataset(self, key: str, group=None):
        with tables.open_file(self.file, "r") as h5:
            if group is not None:
                if group not in h5.root:
                    return False
                return True
            if key in h5.root:
                return True
            else:
                return False

    def _get_group_keys(self, group):
        with tables.open_file(self.file, "r") as h5:  # type: ignore
            if group is not None:
                if group not in h5.root:
                    return []
                slot = h5.get_node(f"/{group}")
            else:
                slot = h5.root
            return list(slot._v_children.keys())


class H5ZSFile(IOGroup):
    INDEX_KEY = "index"
    COORDS_KEY = "coords"
    TABLE_KEY = "table"
    MASKS_KEY = "masks"
    CONTOURS_KEY = "contours"
    HOLES_KEY = "holes"
    TILE_OPS_KEY = "tile_ops"
    FEATURE_FIELDS_KEY = "feature_fields"

    def __init__(self, file: str | Path):
        super().__init__(file)

        self._check_lazyslide_specs()

    def _check_align(self, table):
        data_index = self._read_dataset(self.INDEX_KEY)
        if data_index is None:
            raise RuntimeError("Coordinates must be set before table.")
        assert len(table) == len(
            data_index
        ), "Table length does not match coordinates length"

    def set_coords(self, coords: np.ndarray):
        # Chunk by row-major order
        data_index = np.arange(len(coords))
        self._save_dataset(self.INDEX_KEY, data_index)
        self._save_dataset(self.COORDS_KEY, coords)

    def set_table(self, table: pd.DataFrame):
        self._check_align(table)
        self._save_dataframe(self.TABLE_KEY, table)

    def set_feature_field(self, field: str, value: np.ndarray):
        self._check_align(value)
        self._save_dataset(field, value, group=self.FEATURE_FIELDS_KEY)

    def set_tile_ops(self, tile_ops: TileOps):
        self._save_attributes(self.TILE_OPS_KEY, asdict(tile_ops))

    def set_masks(self, masks: Dict, masks_levels: Dict):
        self._save_attributes(self.MASKS_KEY, masks_levels)
        for name, mask in masks.items():
            self._save_dataset(name, mask, group=self.MASKS_KEY)

    def set_mask(self, name: str, mask: np.ndarray, level: int):
        self._save_dataset(name, mask, group=self.MASKS_KEY)
        mask_levels = self._read_attributes(self.MASKS_KEY)
        if mask_levels is None:
            mask_levels = {}
        mask_levels[name] = level
        self._save_attributes(self.MASKS_KEY, mask_levels)

    def set_contours_holes(self, contours: List[np.ndarray], holes: List[np.ndarray]):
        self._delete_group(self.CONTOURS_KEY)
        self._delete_group(self.HOLES_KEY)

        for i, arr in enumerate(contours):
            self._save_dataset(str(i), arr, group=self.CONTOURS_KEY)
        for i, arr in enumerate(holes):
            self._save_dataset(str(i), arr, group=self.HOLES_KEY)
        self._save_attributes(self.CONTOURS_KEY, {"length": len(contours)})
        self._save_attributes(self.HOLES_KEY, {"length": len(holes)})

    def get_index(self):
        return self._read_dataset(self.INDEX_KEY)

    def get_coords(self):
        return self._read_dataset(self.COORDS_KEY)

    def get_table(self):
        return self._read_dataframe(self.TABLE_KEY)

    def get_tile_ops(self):
        mapping = self._read_attributes(self.TILE_OPS_KEY)
        if mapping is None:
            return None
        return TileOps(**mapping)

    def get_masks(self):
        masks = {}
        masks_levels = self._read_attributes(self.MASKS_KEY)
        for name in masks_levels.keys():
            masks[name] = self._read_dataset(name, group=self.MASKS_KEY)
        return masks, masks_levels

    def get_mask(self, name):
        level = self._read_attributes(self.MASKS_KEY)
        if level is not None:
            level = level[name]
        return (
            self._read_dataset(name, group=self.MASKS_KEY),
            level,
        )

    def get_contours_holes(self):
        contours = []
        holes = []
        for i in range(self._read_attributes(self.CONTOURS_KEY)["length"]):
            contours.append(self._read_dataset(str(i), group=self.CONTOURS_KEY))
        for i in range(self._read_attributes(self.HOLES_KEY)["length"]):
            holes.append(self._read_dataset(str(i), group=self.HOLES_KEY))
        return contours, holes

    def get_feature_field(self, field: str):
        return self._read_dataset(field, group=self.FEATURE_FIELDS_KEY)

    def delete_feature_field(self, field: str):
        self._delete_dataset(field, group=self.FEATURE_FIELDS_KEY)

    def get_available_feature_fields(self):
        return self._get_group_keys(self.FEATURE_FIELDS_KEY)

    def delete_table(self):
        self._delete_dataset(self.TABLE_KEY)

    def has_tiles(self):
        return self._has_dataset(self.COORDS_KEY)
