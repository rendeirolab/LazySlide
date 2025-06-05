import pytest
import lazyslide as zs


@pytest.mark.skip_on_ci
class TestCellSegmentation:
    def test_cell_segmentation(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=0.5, key_added="cell_tiles")

        zs.seg.cells(wsi, "instanseg", tile_key="cell_tiles", key_added="cells")

    def test_cell_classification(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=0.5, key_added="cell_tiles")

        zs.seg.cell_types(wsi, "nulite", tile_key="cell_tiles", key_added="cell_types")


@pytest.mark.skip_on_ci
class TestSemanticSegmentation:
    def test_semantic_segmentation(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=1.5, key_added="semantic_tiles")

        zs.seg.artifact(wsi, tile_key="semantic_tiles")
