__all__ = ["create_reader_store"]


# Adapted from https://github.com/gustaveroussy/sopa/blob/master/sopa/io/reader/wsi.py
from ctypes import ArgumentError
from dataclasses import asdict
from functools import singledispatch
from typing import Dict

import numpy as np
from zarr.storage import (
    _path_to_prefix,
    attrs_key,
    init_array,
    init_group,
    Store,
    KVStore,
)
from zarr.util import json_dumps, normalize_storage_path, normalize_shape

from .reader import TiffSlideReader
from .reader.base import ReaderBase


def init_attrs(store, attrs, path: str = None):
    path = normalize_storage_path(path)
    path = _path_to_prefix(path)
    store[path + attrs_key] = json_dumps(attrs)


def create_meta_store(reader: ReaderBase, tilesize: int) -> Dict[str, bytes]:
    """Creates a dict containing the zarr metadata for the multiscale openslide image."""
    store = dict()
    root_attrs = {
        "multiscales": [
            {
                "name": reader.file,
                "datasets": [
                    {"path": str(i)} for i in range(reader.properties.n_level)
                ],
                "version": "0.1",
            }
        ],
        "metadata": asdict(reader.properties),
    }
    init_group(store)
    init_attrs(store, root_attrs)
    for i, (x, y) in enumerate(reader.properties.level_shape):
        init_array(
            store,
            path=str(i),
            shape=normalize_shape((y, x, 4)),
            chunks=(tilesize, tilesize, 4),
            fill_value=0,
            dtype="|u1",
            compressor=None,
        )
        suffix = str(i) if i != 0 else ""
        init_attrs(
            store, {"_ARRAY_DIMENSIONS": [f"Y{suffix}", f"X{suffix}", "S"]}, path=str(i)
        )
    return store


def _parse_chunk_path(path: str):
    """Returns x,y chunk coords and pyramid level from string key"""
    level, ckey = path.split("/")
    y, x, _ = map(int, ckey.split("."))
    return x, y, int(level)


class ReaderStore(Store):
    """Wraps a Reader object as a multiscale Zarr Store.

    Parameters
    ----------
    reader: Reader
        The reader object
    tilesize: int
        Desired "chunk" size for zarr store (default: 512).
    """

    def __init__(self, reader: ReaderBase, tilesize: int = 512):
        self._reader = reader
        self._tilesize = tilesize
        self._store = create_meta_store(reader, tilesize)
        self._writeable = False
        self._erasable = False

    def __getitem__(self, key: str):
        if key in self._store:
            # key is for metadata
            return self._store[key]

        # key should now be a path to an array chunk
        # e.g '3/4.5.0' -> '<level>/<chunk_key>'
        try:
            x, y, level = _parse_chunk_path(key)
            location = self._ref_pos(x, y, level)
            size = (self._tilesize, self._tilesize)
            tile = self._reader.get_region(*location, *size, level=level)
        except ArgumentError as err:
            # Can occur if trying to read a closed slide
            raise err
        except Exception:
            # TODO: probably need better error handling.
            # If anything goes wrong, we just signal the chunk
            # is missing from the store.
            raise KeyError(key)
        return np.array(tile)

    def __eq__(self, other):
        return (
            isinstance(other, ReaderStore)
            and self._reader.name == other._reader.name
            and self._reader.file == other._reader.file
        )

    def __setitem__(self, key, val):
        raise PermissionError("ZarrStore is read-only")

    def __delitem__(self, key):
        raise PermissionError("ZarrStore is read-only")

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return sum(1 for _ in self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _ref_pos(self, x: int, y: int, level: int):
        level = self._reader.translate_level(level)
        dsample = self._reader.properties.level_downsample[level]
        xref = int(x * dsample * self._tilesize)
        yref = int(y * dsample * self._tilesize)
        return xref, yref

    def keys(self):
        return self._store.keys()

    def close(self):
        self._reader.detach_reader()

    # Retrieved from napari-lazy-openslide PR#16
    def __getstate__(self):
        return {"_path": self._reader.file, "_tilesize": self._tilesize}

    def __setstate__(self, newstate):
        path = newstate["_path"]
        tilesize = newstate["_tilesize"]
        self.__init__(path, tilesize)

    def rename(self, path: str, new_path: str):
        raise PermissionError(f'{type(self)} is not erasable, cannot call "rename"')

    def rmdir(self, path: str = "") -> None:
        raise PermissionError(f'{type(self)} is not erasable, cannot call "rmdir"')


@singledispatch
def create_reader_store(reader: ReaderBase, tilesize: int = 512) -> KVStore:
    """Creates a ReaderStore from a Reader object."""
    return KVStore(ReaderStore(reader, tilesize=tilesize))


@create_reader_store.register(TiffSlideReader)
def _(reader: TiffSlideReader, tilesize: int = 512) -> KVStore:
    return reader.reader.zarr_group.store
