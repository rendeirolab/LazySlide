from mock_models import (
    MockCellSegmentationModel,
    MockCellTypeSegmentationModel,
    MockSemanticSegmentationModel,
)

import lazyslide as zs


def test_tissue_segmentation(wsi):
    zs.seg.tissue(wsi, key_added="tissues")


class TestCellSegmentation:
    def test_cell_segmentation(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=0.5, key_added="cell_tiles")

        model = MockCellSegmentationModel()
        zs.seg.cells(wsi, model, tile_key="cell_tiles", key_added="cells")

    def test_cell_classification(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=0.5, key_added="cell_tiles")

        model = MockCellTypeSegmentationModel()
        zs.seg.cell_types(wsi, model, tile_key="cell_tiles", key_added="cell_types")


class TestSemanticSegmentation:
    def test_semantic_segmentation(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=1.5, key_added="semantic_tiles")

        model = MockSemanticSegmentationModel()
        zs.seg.artifact(wsi, tile_key="semantic_tiles", model=model)
