from lazyslide.readers.base import ReaderBase
from lazyslide.readers.vips import VipsReader


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
