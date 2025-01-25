"""Efficient and Scalable Whole Slide Image (WSI) processing library."""
__version__ = "0.3.0"


from . import preprocess as pp
from . import tools as tl
from . import plotting as pl
from . import segmentation as seg
from . import cv as cv
from . import models


# Re-export the public API
from wsidata import open_wsi, agg_wsi
