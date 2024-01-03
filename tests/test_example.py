import pytest

from conftest import import_windows_modules

import_windows_modules()


import lazyslide as zs


@pytest.mark.parametrize("reader", ["vips", "openslide"])
def test_example(reader):
    slide = "https://github.com/camicroscope/Distro/raw/master/images/sample.svs"
    wsi = zs.WSI(slide, reader=reader)
    wsi.create_tissue_mask(threshold=7, max_hole_size=500)
    wsi.create_tiles(256)

    del wsi
    # reinitialize
    wsi = zs.WSI(slide, reader=reader)

    # delete the h5 file
    import os

    os.remove(wsi.h5_file.file)
