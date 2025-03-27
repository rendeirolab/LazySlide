import pytest

import lazyslide as zs


@pytest.mark.parametrize("detect_holes", [True, False])
@pytest.mark.parametrize("key_added", ["tissue", "tissue2"])
def test_pp_find_tissues(wsi, detect_holes, key_added):
    zs.pp.find_tissues(wsi, detect_holes=detect_holes, key_added=key_added)

    assert key_added in wsi.shapes
    if not detect_holes:
        tissue = wsi[key_added].geometry[0]
        assert len(tissue.interiors) == 0


class TestPPTileTissues:
    def test_tile_px(self, wsi):
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles")

    def test_mpp(self, wsi):
        zs.pp.tile_tissues(wsi, 256, mpp=1, key_added="tiles1")

    @pytest.mark.xfail(raises=ValueError)
    def test_slide_mpp(self, wsi):
        zs.pp.tile_tissues(wsi, 256, slide_mpp=1, key_added="tiles2")

    def test_assert(self, wsi):
        s0 = len(wsi.sdata["tiles"])
        s1 = len(wsi.sdata["tiles1"])

        assert s0 > 0
        assert s1 < s0
