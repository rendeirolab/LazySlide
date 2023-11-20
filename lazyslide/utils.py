from lazyslide.backends.base import BackendBase
from lazyslide.backends.vips import VipsBackend


def get_backend() -> BackendBase:
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
        return VipsBackend

