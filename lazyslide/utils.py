from dataclasses import dataclass, field

from .readers.base import ReaderBase
from .readers.vips import VipsReader


def get_reader() -> ReaderBase:
    """Return an available backend"""
    pyvips_avail = False
    cucim_avail = False
    openslide_avail = False

    try:
        import pyvips as vips
        pyvips_avail = True
    except (ModuleNotFoundError, OSError) as e:
        pass

    if pyvips_avail:
        return VipsReader

    else:
        raise RuntimeError("Cannot find a suitable image reader")


@dataclass
class TileOps:
    level: int
    downsample: float
    mpp: float = field(default=None)
    height: int = field(default=None)
    width: int = field(default=None)
    ops_height: int = field(default=None)
    ops_width: int = field(default=None)
    mask_name: str = field(default=None)