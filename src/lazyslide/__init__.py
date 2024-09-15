"""Efficient and Scalable Whole Slide Image (WSI) processing library."""
__version__ = "0.1.0"

from . import preprocess as pp
from . import tools as tl
from . import plotting as pl
from . import models as models

# Re-export
from wsidata import open_wsi, agg_wsi
