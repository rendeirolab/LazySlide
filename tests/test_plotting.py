from lazyslide import WSI


def test_plot_tissue():
    """Test plot_tissue."""
    wsi = WSI("../data/Artery-with-holes.svs")
    wsi.plot_tissue(tiles=True)
