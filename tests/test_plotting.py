import numpy as np
import pandas as pd

from lazyslide import WSI


class TestPlotting:
    def setup_method(self):
        wsi = WSI("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        wsi.create_tissue_mask()
        wsi.create_tissue_contours()
        wsi.create_tiles(512)
        self.wsi = wsi

    def test_plot_tissue(self):
        """Test plot_tissue."""
        wsi = self.wsi
        wsi.plot_tissue()

    def test_plot_tissue_mask(self):
        """Test plot_tissue_mask."""
        wsi = self.wsi
        wsi.plot_mask()

    def test_plot_tissue_contours(self):
        """Test plot_tissue_contours."""
        wsi = self.wsi
        wsi.plot_tissue(contours=True)

    def test_plot_table(self):
        """Test plot_table."""
        wsi = self.wsi
        wsi.new_table(
            pd.DataFrame(
                {
                    "num": np.arange(wsi.n_tiles),
                    "cat": np.random.choice(["a", "b"], wsi.n_tiles),
                }
            )
        )
        wsi.plot_table("num")
        wsi.plot_table("cat")
        wsi.plot_table(["num", "cat"])

    def test_plot_feature(self):
        """Test plot_features."""
        wsi = self.wsi
        wsi.new_feature("feature", np.random.rand(wsi.n_tiles, 10))
        wsi.plot_feature("feature", index=4)
        wsi.plot_feature("feature", index=[0, 1, 4, 5])
