from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from itertools import tee
from typing import Type
from urllib.parse import urlparse

import requests

from .readers.base import ReaderBase
from .readers.vips import VipsReader
from .readers.openslide import OpenSlideReader


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

    # try:
    #     import cucim
    #     readers["cucim"] = CuCIMReader
    # except (ModuleNotFoundError, OSError) as _:
    #     pass
    reader_candidates = ["cucim", "vips", "openslide"]
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
    raise ValueError("Path must be a URL or Path to existing file.")


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
