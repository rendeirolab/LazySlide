import geopandas as gpd
from shapely import box
from wsidata.io import add_shapes

import lazyslide as zs


def test_tile_shaper_merges_adjacent_tiles(wsi_no_spec):
    """Regression: tile_shaper raised AttributeError on every call."""
    tiles = gpd.GeoDataFrame(
        {
            "tissue_id": 0,
            "domain": ["a", "a", "b", "b"],
            "geometry": [
                box(0, 0, 10, 10),
                box(10, 0, 20, 10),
                box(20, 0, 30, 10),
                box(50, 0, 60, 10),
            ],
        }
    )
    add_shapes(wsi_no_spec, "domain_tiles", tiles)
    zs.tl.tile_shaper(wsi_no_spec, groupby="domain", tile_key="domain_tiles")

    shapes = wsi_no_spec["domain_shapes"]
    # The two "a" tiles touch and merge; the "b" tiles are apart
    assert sorted(zip(shapes["domain"], shapes.area, strict=True)) == [
        ("a", 200.0),
        ("b", 100.0),
        ("b", 100.0),
    ]
