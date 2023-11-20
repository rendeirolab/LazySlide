from lazyslide.backends.base import BackendBase


def get_backend() -> BackendBase:
    """Return an available backend"""
    try:
        import pyvips as vips
    except (ModuleNotFoundError, OSError) as e:
        log.error("Unable to load vips; slide processing will be unavailable. "
                  f"Error raised: {e}")