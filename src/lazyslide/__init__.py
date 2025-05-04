"""Efficient and Scalable Whole Slide Image (WSI) processing library."""

__version__ = "0.6.0"


import sys

# Re-export the public API
from wsidata import open_wsi, agg_wsi

from . import cv
from . import io
from . import models
from . import plotting as pl
from . import preprocess as pp
from . import segmentation as seg
from . import tools as tl
from . import datasets
from . import metrics

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
]
