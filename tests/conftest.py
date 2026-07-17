import os

import pytest
import torch
from torch.export import dims, export, save

# When set (e.g. on fork PRs without HF secrets), skip dataset download and fixtures.
SKIP_DATASET_TESTS = os.environ.get("LAZYSLIDE_SKIP_DATASET_TESTS") == "1"


class MockNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.shape[0], 1000)


def cache_test_datasets():
    import lazyslide as zs

    zs.datasets.gtex_artery()
    zs.datasets.sample()


def pytest_sessionstart(session):
    if not SKIP_DATASET_TESTS:
        cache_test_datasets()


@pytest.fixture(scope="session")
def wsi():
    if SKIP_DATASET_TESTS:
        pytest.skip("Dataset tests skipped (no HF token, e.g. fork PR)")
    import lazyslide as zs

    return zs.datasets.gtex_artery()


@pytest.fixture(scope="class")
def wsi_small():
    if SKIP_DATASET_TESTS:
        pytest.skip("Dataset tests skipped (no HF token, e.g. fork PR)")
    import lazyslide as zs

    return zs.datasets.sample()


@pytest.fixture(scope="session")
def wsi_no_spec():
    """WSI with arbitrary shapes as tiles — no TileSpec metadata."""
    if SKIP_DATASET_TESTS:
        pytest.skip("Dataset tests skipped (no HF token, e.g. fork PR)")
    import geopandas as gpd
    from shapely import box
    from wsidata.io import add_shapes

    import lazyslide as zs

    wsi = zs.datasets.sample(with_data=False)
    tiles = [
        box(100, 100, 356, 356),
        box(500, 200, 756, 456),
        box(100, 600, 356, 856),
        box(800, 800, 1056, 1056),
        box(400, 1200, 656, 1456),
    ]
    gdf = gpd.GeoDataFrame({"geometry": tiles})
    add_shapes(wsi, "no_spec_tiles", gdf)
    return wsi


@pytest.fixture(scope="session")
def tmp_path_session(tmp_path_factory):
    return tmp_path_factory.mktemp("session_tmp")


@pytest.fixture(scope="session")
def torch_model_file(tmp_path_session):
    model = MockNet()
    torch.save(model, tmp_path_session / "model.pt")
    return tmp_path_session / "model.pt"


@pytest.fixture(scope="session")
def torch_jit_file(tmp_path_session):
    model = MockNet()
    torch.jit.script(model).save(tmp_path_session / "jit_model.pt")
    return tmp_path_session / "jit_model.pt"


@pytest.fixture(scope="session")
def torch_export_model(tmp_path_session):
    model = MockNet()

    batch_dim = dims("batch")
    exp_mods = export(
        model, args=torch.randn(1, 3, 224, 224), dynamic_shapes={0: batch_dim}
    )
    save(exp_mods, tmp_path_session / "exported_model.pt2")


@pytest.fixture(scope="session")
def wsi_with_annotations():
    """Fixture that provides a WSI with annotations for testing."""
    if SKIP_DATASET_TESTS:
        pytest.skip("Dataset tests skipped (no HF token, e.g. fork PR)")
    import geopandas as gpd
    from shapely.geometry import Polygon

    import lazyslide as zs

    wsi = zs.datasets.gtex_artery()

    # Create some simple annotations if they don't exist
    if "annotations" not in wsi.shapes:
        # Get a tissue polygon to use as an annotation
        if len(wsi["tissues"]) > 0:
            tissue_poly = wsi["tissues"].geometry.iloc[0]
            # Create a smaller polygon inside the tissue
            minx, miny, maxx, maxy = tissue_poly.bounds
            width, height = maxx - minx, maxy - miny

            # Create a polygon that's 25% of the size in the center
            small_poly = [
                (minx + width * 0.375, miny + height * 0.375),
                (minx + width * 0.625, miny + height * 0.375),
                (minx + width * 0.625, miny + height * 0.625),
                (minx + width * 0.375, miny + height * 0.625),
            ]

            # Add to wsi as annotations
            annot_gdf = gpd.GeoDataFrame(
                {"geometry": [Polygon(small_poly)], "class": ["test_annotation"]}
            )
            wsi.shapes["annotations"] = annot_gdf

    return wsi
