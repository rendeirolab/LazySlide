import pytest
import lazyslide as zs


class TestIO:
    @pytest.mark.parametrize("reader", ["openslide", "tiffslide"])
    def test_local_file(self, reader):
        file = "data/CMU-1-Small-Region.svs"
        zs.WSI(file, reader=reader)

    @pytest.mark.parametrize("reader", ["openslide", "tiffslide"])
    def test_remote_file(self, reader):
        file = "https://github.com/camicroscope/Distro/raw/master/images/sample.svs"
        zs.WSI(file, reader=reader)

    # A test save to pytest temporary file
    def test_save_temp(self, tmp_path):
        file = "data/CMU-1-Small-Region.svs"
        wsi = zs.WSI(file)
        wsi.write(tmp_path / "test.zarr")
