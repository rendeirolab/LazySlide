import numpy as np

import lazyslide as zs

from .mock_models import (
    MockCellSegmentationModel,
    MockCellTypeSegmentationModel,
    MockSemanticSegmentationModel,
)


def _ref_pool_cell_feature(instance_map, patch_token_map, instance_id):
    """Reference: original single-instance pooling, used to pin the behaviour of
    the vectorized :func:`_pool_cell_features`."""
    cell_mask = instance_map == instance_id
    H, W = cell_mask.shape
    D, PH, PW = patch_token_map.shape
    row_idx = np.round(np.linspace(0, H - 1, PH)).astype(int)
    col_idx = np.round(np.linspace(0, W - 1, PW)).astype(int)
    patch_mask = cell_mask[np.ix_(row_idx, col_idx)]
    if patch_mask.any():
        return patch_token_map[:, patch_mask].mean(axis=1)
    ys, xs = np.where(cell_mask)
    py = min(int(ys.mean() * PH / H), PH - 1)
    px = min(int(xs.mean() * PW / W), PW - 1)
    return patch_token_map[:, py, px]


def test_pool_cell_features_matches_reference():
    """The batched/vectorized per-cell token pooling must match the original
    per-instance implementation exactly, including the small-cell fallback path
    (a cell too small to land on the downsampled patch grid)."""
    from lazyslide.segmentation._seg_runner import _pool_cell_features

    rng = np.random.RandomState(0)
    H = W = 128
    D, PH, PW = 16, 8, 8
    tok = rng.randn(D, PH, PW).astype(np.float32)
    im = np.zeros((H, W), dtype=np.int32)
    im[0:40, 0:40] = 3
    im[50:90, 60:100] = 7
    im[100:120, 10:30] = 11
    im[40:42, 45:47] = 99  # tiny instance between grid points -> fallback path
    ids = [3, 7, 11, 99]

    out = _pool_cell_features(im, tok, ids)
    assert set(out) == set(ids)
    for cid in ids:
        ref = _ref_pool_cell_feature(im, tok, cid)
        assert out[cid].dtype == np.float32
        assert np.allclose(out[cid], ref, atol=1e-6)
    # Empty input is handled.
    assert _pool_cell_features(im, tok, []) == {}


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
        # Overlapping tiles: the exact configuration that previously produced a
        # shapes/features count mismatch — MultiPolygon cells were exploded into
        # extra shape rows while features stayed one-per-cell.
        zs.pp.tile_tissues(
            wsi, tile_px=512, mpp=0.5, overlap=0.5, key_added="cell_feat_tiles"
        )

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

        shapes = wsi.shapes["cell_types_feat"]
        n_rows = len(shapes)
        assert n_rows > 0, "Expected deterministic mock segmentation to produce cells"
        # cell_id column present + every row has a non-null id
        assert "cell_id" in shapes.columns
        assert shapes["cell_id"].notna().all()
        # One row per cell: cell_id is unique even though some cells are
        # MultiPolygons (no explode inflation).
        assert shapes["cell_id"].is_unique
        # The mock emits a MultiPolygon cell; it must survive as a single row.
        assert (shapes.geometry.geom_type == "MultiPolygon").any()

        # Verify features were stored
        feat_key = "cell_types_feat_features"
        assert feat_key in wsi.tables
        feat = wsi.tables[feat_key]
        assert feat.X.shape[1] == model._EMBED_DIM

        # cell_id is the join key — shapes and features are strictly 1:1.
        assert "cell_id" in feat.obs.columns
        shape_ids = set(shapes["cell_id"].astype(int).tolist())
        feat_ids = set(feat.obs["cell_id"].astype(int).tolist())
        assert shape_ids == feat_ids
        # Exactly one shape row and one feature row per cell.
        assert feat.n_obs == n_rows
        assert feat.n_obs == len(shape_ids)

    def test_cell_segmentation_multipolygon_alignment(self, wsi):
        """Regression: a cell whose geometry is a MultiPolygon (e.g. ``buffer(0)``
        split a pinched mask into disjoint parts) must stay a SINGLE shape row,
        1:1 with its single feature row. ``explode()`` used to expand such cells
        into multiple shape rows, leaving the shape and feature counts mismatched
        (the bug reproduced with overlapping tiles)."""
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=0.5, key_added="mp_tiles")
        model = MockCellSegmentationModel(emit_tokens=True)
        zs.seg.cells(
            wsi,
            model,
            tile_key="mp_tiles",
            key_added="mp_cells",
            extract_features=True,
        )
        shapes = wsi.shapes["mp_cells"]
        feat = wsi.tables["mp_cells_features"]
        assert "cell_id" in shapes.columns
        assert "cell_id" in feat.obs.columns
        # The mock emits a corner-touching cell that becomes a MultiPolygon;
        # it must be present and kept as a single row.
        assert (shapes.geometry.geom_type == "MultiPolygon").any()
        assert shapes["cell_id"].is_unique
        # Shapes and features are strictly 1:1.
        assert len(shapes) == feat.n_obs
        assert set(shapes["cell_id"].astype(int)) == set(
            feat.obs["cell_id"].astype(int).tolist()
        )


class TestSemanticSegmentation:
    def test_semantic_segmentation(self, wsi):
        zs.pp.tile_tissues(wsi, tile_px=512, mpp=1.5, key_added="semantic_tiles")

        model = MockSemanticSegmentationModel()
        zs.seg.artifact(wsi, tile_key="semantic_tiles", model=model)
