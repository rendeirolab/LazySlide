import pytest
import lazyslide as zs


class TestIO:
    @pytest.mark.parametrize("reader", ["openslide", "tiffslide"])
    def test_local_file(self, reader, test_slide):
        zs.WSI(test_slide, reader=reader)

    @pytest.mark.parametrize("reader", ["openslide", "tiffslide"])
    def test_remote_file(self, reader):
        file = "https://github.com/camicroscope/Distro/raw/master/images/sample.svs"
        zs.WSI(file, reader=reader)

    # A test save to pytest temporary file
    def test_save_temp(self, tmp_path, test_slide):
        wsi = zs.WSI(test_slide)
        wsi.write(tmp_path / "test.zarr")
