"""Efficient and Scalable Whole Slide Image (WSI) processing library."""

from ._version import version

__version__ = version

import sys

# Re-export the public API from wsidata
from wsidata import agg_wsi, open_wsi

from . import cv, datasets, io, metrics
from . import plotting as pl
from . import preprocess as pp
from . import segmentation as seg
from . import tools as tl
from ._setting import settings

# Inject the aliases into the current module
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl", "seg"]})
del sys


def __getattr__(name):
    # Lazy-load deprecated `models` submodule so FutureWarning + heavy deps
    # only trigger when user actually accesses `lazyslide.models`.
    if name == "models":
        import importlib

        mod = importlib.import_module(".models", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["models"])


__all__ = [
    "open_wsi",
    "agg_wsi",
    "pp",
    "tl",
    "pl",
    "seg",
    "cv",
    "models",
    "io",
    "datasets",
    "metrics",
    "settings",
]
