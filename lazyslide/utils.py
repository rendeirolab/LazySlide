from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from itertools import tee
from typing import Type
from urllib.parse import urlparse

import numpy as np
from numba import njit

import requests

from .readers.base import ReaderBase
from .readers import VipsReader, OpenSlideReader, CuCIMReader


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_reader(reader="auto") -> Type[ReaderBase]:
    """Return an available backend"""

    readers = {"openslide": None, "vips": None, "cucim": None}

    try:
        import openslide

        readers["openslide"] = OpenSlideReader
    except (ModuleNotFoundError, OSError) as _:
        pass

    try:
        import pyvips as vips

        readers["vips"] = VipsReader
    except (ModuleNotFoundError, OSError) as _:
        pass

    try:
        import cucim

        readers["cucim"] = CuCIMReader
    except (ModuleNotFoundError, OSError) as _:
        pass

    reader_candidates = ["openslide", "cucim", "vips"]
    if reader == "auto":
        for i in reader_candidates:
            reader = readers.get(i)
            if reader is not None:
                return reader
    elif reader not in reader_candidates:
        raise ValueError(
            f"Reqeusted reader not available, " f"must be one of {reader_candidates}"
        )
    else:
        return readers[reader]


def is_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def download_file(url: str, file_path: Path, chunk_size: int = 1024):
    """Download a file in chunks"""
    r = requests.get(url, stream=True)
    with file_path.open("wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def check_wsi_path(path: str | Path, allow_download: bool = True) -> Path:
    import tempfile

    # Check path is URL or Path
    if isinstance(path, str):
        if is_url(path):
            if not allow_download:
                raise ValueError("Given a URL but not allowed to download.")
            file_path = Path(tempfile.mkdtemp()) / path.split("/")[-1].split("?")[0]
            download_file(path, file_path)
            return file_path
        elif Path(path).exists():
            return Path(path)
    elif isinstance(path, Path):
        if path.exists():
            return path
    raise ValueError(f"{path} not exists.")


@dataclass
class TileOps:
    level: int = 0
    downsample: float = 1
    mpp: float = field(default=None)
    height: int = field(default=None)
    width: int = field(default=None)
    ops_height: int = field(default=None)
    ops_width: int = field(default=None)
    mask_name: str = field(default=None)


@njit
def _creat_tiles_params(image_shape, tile_w, tile_h, stride_w=None, stride_h=None, pad=True):
    height, width = image_shape
    if stride_w is None:
        stride_w = tile_w
    if stride_h is None:
        stride_h = tile_h

    # calculate number of expected tiles
    if pad and width % stride_w != 0:
        n_tiles_width = width // stride_w + 1
    else:
        n_tiles_width = (width - tile_w) // stride_w + 1
    if pad and height % stride_h != 0:
        n_tiles_height = height // stride_h + 1
    else:
        n_tiles_height = (height - tile_h) // stride_h + 1
    return n_tiles_width, n_tiles_height


@njit
def create_tiles(
        image_shape, tile_w, tile_h, stride_w=None, stride_h=None, pad=True
):
    """Create the tiles, return only coordination

    Padding works as follows:
    If ``pad is False``, then the first tile will start flush with the edge of the image, and the tile locations
    will increment according to specified stride, stopping with the last tile that is fully contained in the image.
    If ``pad is True``, then the first tile will start flush with the edge of the image, and the tile locations
    will increment according to specified stride, stopping with the last tile which starts in the image. Regions
    outside the image will be padded with 0.
    For example, for a 5x5 image with a tile size of 3 and a stride of 2, tile generation with ``pad=False`` will
    create 4 tiles total, compared to 6 tiles if ``pad=True``.

    Parameters
    ----------
    image_shape : (int, int), The shape of the image
    tile_w : int, The width of tile
    tile_h : int, The height of tile
    stride_w : int, The width of stride when move to next tile
    stride_h : int, The height of stride when move to next tile
    pad : bool, If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks.
                If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.

    Returns
    -------

    """
    n_tiles_width, n_tiles_height = _creat_tiles_params(image_shape, tile_w, tile_h, stride_w, stride_h, pad)

    coordinates = []
    for ix_width in range(n_tiles_width):
        for ix_height in range(n_tiles_height):
            coords = (int(ix_width * stride_w), int(ix_height * stride_h))
            coordinates.append(coords)

    return np.array(coordinates, dtype=np.uint)


@njit
def create_tiles_coords_index(
        image_shape, tile_w, tile_h, stride_w=None, stride_h=None, pad=True
):
    """Create the tiles, return coordination that comprise the tiles
        and the index of points for each rect

    Padding works as follows:
    If ``pad is False``, then the first tile will start flush with the edge of the image, and the tile locations
    will increment according to specified stride, stopping with the last tile that is fully contained in the image.
    If ``pad is True``, then the first tile will start flush with the edge of the image, and the tile locations
    will increment according to specified stride, stopping with the last tile which starts in the image. Regions
    outside the image will be padded with 0.
    For example, for a 5x5 image with a tile size of 3 and a stride of 2, tile generation with ``pad=False`` will
    create 4 tiles total, compared to 6 tiles if ``pad=True``.

    Parameters
    ----------
    image_shape : (int, int), The shape of the image
    tile_h : int, The height of tile
    tile_w : int, The width of tile
    stride_h : int, The height of stride when move to next tile
    stride_w : int, The width of stride when move to next tile
    pad : bool, If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks.
                If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.

    Returns
    -------

    """
    n_tiles_width, n_tiles_height = _creat_tiles_params(image_shape, tile_w, tile_h, stride_w, stride_h, pad)

    coordinates = list()
    indices = list()
    for ix_width in range(n_tiles_width + 1):
        for ix_height in range(n_tiles_height + 1):
            coords = (int(ix_width * stride_w), int(ix_height * stride_h))
            coordinates.append(coords)
            if (ix_width != n_tiles_width) & (ix_height != n_tiles_height):
                ix1 = ix_height * (n_tiles_width + 1) + ix_width
                ix3 = (ix_height + 1) * (n_tiles_width + 1) + ix_width
                indices.append([ix1, ix1 + 1, ix3 + 1, ix3])

    return np.array(coordinates, dtype=np.uint), np.array(indices, dtype=np.uint)


@njit
def filter_tiles(mask, tiles_coords, tile_w, tile_h, filter_bg=0.8):
    """Return a binary array that indicate which tile should be left

    Parameters
    ----------
    mask
    tiles_coords
    filter_bg
    tile_w,
    tile_h,
    Returns
    -------

    """
    use = []
    for w, h in tiles_coords:
        mask_region = mask[h: h + tile_h, w: w + tile_w]
        bg_ratio = np.sum(mask_region == 0) / mask_region.size
        use.append(bg_ratio < filter_bg)
    return np.array(use, dtype=np.bool_)


def get_split_image_indices(image_height, image_width, min_side=20000):
    h, w = image_height, image_width
    size = h * w
    n = min_side
    if (size > n ** 2) or (h > n) or (w > n):
        split_h = h > 1.5 * n
        split_w = w > 1.5 * n

        if not split_h and not split_w:
            return

        n_chunk_h = int(np.ceil(h / n))
        n_chunk_w = int(np.ceil(w / n))

        # If split, return the split chunks
        # Else, it would take the whole
        ix_h = (
            np.linspace(start=0, stop=h, num=n_chunk_h + 1, dtype=int)
            if split_h
            else [0, h]
        )
        ix_w = (
            np.linspace(start=0, stop=w, num=n_chunk_w + 1, dtype=int)
            if split_w
            else [0, w]
        )

        slices = []
        for h1, h2 in pairwise(ix_h):
            row = []
            for w1, w2 in pairwise(ix_w):
                row.append((h1, h2, w1, w2))
            slices.append(row)
        return slices


@njit
def create_mesh_array(shape, coords, h, w, values=None):
    """Create a mesh array from a given image and coordinates

    Parameters
    ----------
    img : np.ndarray
        The image
    coords : np.ndarray
        The coordinates
    h : int
        The height of the tiles
    w : int
        The width of the tiles
    values : np.ndarray
        The values to fill in the mesh array

    Returns
    -------
    np.ndarray
        The mesh array
    """
    mesh = np.full(shape, fill_value=np.nan)
    if values is None:
        for cw, ch in coords:
            mesh[int(ch):int(ch + h), int(cw):int(cw + w)] = 1
    else:
        for (cw, ch), v in zip(coords, values):
            mesh[int(ch):int(ch + h), int(cw):int(cw + w)] = v
    return mesh
