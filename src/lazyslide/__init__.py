"""Efficient and Scalable Whole Slide Image (WSI) processing library."""

from ._version import version

__version__ = version

import sys

# Re-export the public API from wsidata
from wsidata import agg_wsi, open_wsi

from lazyslide._lazy import LazyLoader as _LazyLoader

# Create lazy loaders for submodules
cv = _LazyLoader("lazyslide.cv")
datasets = _LazyLoader("lazyslide.datasets")
io = _LazyLoader("lazyslide.io")
metrics = _LazyLoader("lazyslide.metrics")
models = _LazyLoader("lazyslide.models")
pl = _LazyLoader("lazyslide.plotting")
pp = _LazyLoader("lazyslide.preprocess")
seg = _LazyLoader("lazyslide.segmentation")
tl = _LazyLoader("lazyslide.tools")

# Inject the aliases into the current module
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl", "seg"]})
del sys


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
]
