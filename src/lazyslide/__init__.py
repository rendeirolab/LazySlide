"""Efficient and Scalable Whole Slide Image (WSI) processing library."""

from ._version import version

__version__ = version

import sys

# Re-export the public API from wsidata
from wsidata import agg_wsi, open_wsi

from . import cv, datasets, io, metrics, models
from . import plotting as pl
from . import preprocess as pp
from . import segmentation as seg
from . import tools as tl

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
