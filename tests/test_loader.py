from conftest import import_windows_modules

import_windows_modules()

import lazyslide as zs
from lazyslide.loader import SlidesBalancedLoader


def test_balanced_loader():
    slide = "https://github.com/camicroscope/Distro/raw/master/images/sample.svs"
    wsi = zs.WSI(slide)
    wsi.create_tissue_mask()
    wsi.create_tiles(512)
    loader = SlidesBalancedLoader(wsi_list=[wsi], shared_memory=True)

    for _ in zip(loader, range(10)):
        break
