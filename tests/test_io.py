import lazyslide as zs


class TestIO:
    def test_local_file(self):
        file = "data/CMU-1-Small-Region.svs"
        zs.WSI(file)

    def test_remote_file(self):
        file = "https://github.com/camicroscope/Distro/raw/master/images/sample.svs"
        zs.WSI(file)

    # A test save to pytest temporary file
    def test_save_temp(self, tmp_path):
        file = "data/CMU-1-Small-Region.svs"
        wsi = zs.WSI(file)
        wsi.write(tmp_path / "test.zarr")
