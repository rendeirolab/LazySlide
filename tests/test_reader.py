from functools import partial

import numpy as np

from lazyslide import WSI

import pytest


@pytest.mark.parametrize("reader", ["openslide", "tiffreader"])
def test_read_slide(reader):
    slide = "https://github.com/camicroscope/Distro/raw/master/images/sample.svs"
    wsi = WSI(slide, reader=reader)
    wrapper_read = wsi.get_patch(500, 600, 100, 150, 0)
    # remove alpha channel
    openslide_read = np.array(wsi.reader.slide.read_region((500, 600), 0, (100, 150)))[
        ..., :3
    ]
    assert np.array_equal(wrapper_read, openslide_read)
