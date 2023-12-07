"""Working with whole slide imaging"""
__version__ = "0.1.0"

from .wsi import WSI
from .utils import get_reader
from .models import CTransPath, RetCCL, HoVerNet
from .loader import FeatureExtractionDataset, SlidesBalancedLoader


def about():
    """Provide current information for the Lazyslide"""
    print("Version:", __version__)
    print("Backend:", get_reader().__name__)
