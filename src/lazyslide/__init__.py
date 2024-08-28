import pkg_resources

from wsi_data import open_wsi
import lazyslide.pp as pp
import lazyslide.tl as tl
import lazyslide.pl as pl

version = __version__ = pkg_resources.get_distribution("lazyslide").version
