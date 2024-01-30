import pytest

from conftest import import_windows_modules

import_windows_modules()

import lazyslide as zs


@pytest.mark.parametrize("reader", ["openslide"])
class TestWSI:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self, reader):
        slide = "https://github.com/camicroscope/Distro/raw/master/images/sample.svs"
        self.slide = slide
        self.wsi = zs.WSI(slide, reader=reader)

    def test_create_mask(self, reader):
        wsi = self.wsi
        wsi.create_tissue_mask(threshold=7, max_hole_size=500)
        wsi.create_tiles(256, mpp=0.5)

    def test_create_contours(self, reader):
        wsi = self.wsi
        wsi.create_tissue_contours()
        wsi.create_tiles((250, 280), (10, 10))

    def test_reinitialize(self, reader):
        del self.wsi
        # reinitialize
        zs.WSI(self.slide)
