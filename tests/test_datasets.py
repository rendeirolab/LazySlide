import lazyslide as zs


def test_load_sample():
    wsi = zs.datasets.sample()
    assert wsi is not None


def test_load_gtex_artery():
    wsi = zs.datasets.gtex_artery()
    assert wsi is not None


def test_load_lung_carcinoma():
    wsi = zs.datasets.lung_carcinoma()
    assert wsi is not None
