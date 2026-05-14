import lazyslide as zs

from .mock_models import (
    MockCellSegmentationModel,
    MockCellTypeSegmentationModel,
    MockSemanticSegmentationModel,
)


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

    def test_cell_classification_with_features(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=0.5, key_added="cell_feat_tiles")

        model = MockCellTypeSegmentationModel()
        zs.seg.cell_types(
            wsi,
            model,
            tile_key="cell_feat_tiles",
            key_added="cell_types_feat",
            extract_features=True,
        )
        # Verify cells were added
        assert "cell_types_feat" in wsi.shapes

        n_cells = len(wsi.shapes["cell_types_feat"])
        if n_cells > 0:
            # Verify features were stored
            feat_key = "cell_types_feat_features"
            assert feat_key in wsi.tables
            feat = wsi.tables[feat_key]
            assert feat.X.shape[0] == n_cells
            assert feat.X.shape[1] == model._EMBED_DIM


class TestSemanticSegmentation:
    def test_semantic_segmentation(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=1.5, key_added="semantic_tiles")

        model = MockSemanticSegmentationModel()
        zs.seg.artifact(wsi, tile_key="semantic_tiles", model=model)
