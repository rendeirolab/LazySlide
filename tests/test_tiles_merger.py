"""Unit tests for non-maximum suppression used to merge overlapping-tile cells."""

import geopandas as gpd
from shapely import Polygon, box

from lazyslide.cv import nms


def test_nms_keeps_nonoverlapping():
    gdf = gpd.GeoDataFrame(
        {
            "prob": [0.5, 0.6, 0.7],
            "id": [10, 11, 12],
            "geometry": [
                box(0, 0, 10, 10),
                box(100, 100, 110, 110),
                box(200, 200, 210, 210),
            ],
        }
    )
    out = nms(gdf, "prob")
    assert set(out["id"]) == {10, 11, 12}


def test_nms_suppresses_overlapping_keeps_highest_prob():
    # Two coincident boxes (IoU == 1) -> keep only the higher-probability one.
    gdf = gpd.GeoDataFrame(
        {
            "prob": [0.2, 0.9],
            "id": [1, 2],
            "geometry": [box(0, 0, 10, 10), box(0, 0, 10, 10)],
        }
    )
    out = nms(gdf, "prob")
    assert list(out["id"]) == [2]


def test_nms_drops_invalid_without_misaligning():
    """Regression: ``preprocess_gdf`` drops invalid/empty geometries, so the
    STRtree is built over a subset of rows. nms must map tree positions back to
    the correct original rows. The previous implementation indexed ``gdf`` by
    tree positions directly and returned the wrong rows once anything was
    dropped."""
    g_valid_a = box(0, 0, 10, 10)
    # Zero-area "polygon" -> invalid, becomes empty after buffer(0) -> dropped.
    g_degenerate = Polygon([(100, 100), (110, 100), (100, 100)])
    g_valid_b = box(200, 200, 210, 210)
    gdf = gpd.GeoDataFrame(
        {
            "prob": [0.5, 0.9, 0.7],
            "id": [0, 1, 2],
            "geometry": [g_valid_a, g_degenerate, g_valid_b],
        }
    )
    out = nms(gdf, "prob")
    # The degenerate row (id=1) is dropped; the two real, non-overlapping boxes
    # survive with their correct ids — not the row that happens to sit at the
    # same tree position.
    assert set(out["id"]) == {0, 2}
    # Geometry stays paired with its id (no positional drift).
    by_id = dict(zip(out["id"], out.geometry))
    assert by_id[0].equals(g_valid_a)
    assert by_id[2].equals(g_valid_b)
