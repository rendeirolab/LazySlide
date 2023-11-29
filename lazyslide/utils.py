from dataclasses import dataclass, field
from typing import Type

from .readers.base import ReaderBase
from .readers.vips import VipsReader
from .readers.openslide import OpenSlideReader


def get_reader(reader="auto") -> Type[ReaderBase]:
    """Return an available backend"""

    readers = {
        "openslide": None,
        "vips": None,
        "cucim": None
    }

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
        raise ValueError(f"Reqeusted reader not available, "
                         f"must be one of {reader_candidates}")
    else:
        return readers[reader]


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
