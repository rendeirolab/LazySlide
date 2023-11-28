"""Working with whole slide imaging"""
__version__ = "0.1.0"

from .wsi import WSI, create_tiles_coords, filter_tiles
from .readers.utils import get_crop_left_top_width_height
from .cv_mods import TissueDetectionHE
from .h5 import H5File
