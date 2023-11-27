"""Working with whole slide imaging"""
__version__ = "0.1.0"

from .wsi import WSI
from .readers.utils import get_crop_xy_wh
from .cv_mods import TissueDetectionHE
from .h5 import H5File
