import os

import pytest
import torch


class MockNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.shape[0], 1000)


@pytest.fixture(scope="session", autouse=True)
def wsi():
    import lazyslide as zs

    return zs.datasets.gtex_artery()


@pytest.fixture(scope="session", autouse=True)
def wsi_small():
    import lazyslide as zs

    return zs.datasets.sample()


@pytest.fixture(scope="session")
def tmp_path_session(tmp_path_factory):
    return tmp_path_factory.mktemp("session_tmp")


@pytest.fixture(scope="session", autouse=True)
def torch_model_file(tmp_path_session):
    model = MockNet()
    torch.save(model, tmp_path_session / "model.pt")
    return tmp_path_session / "model.pt"


@pytest.fixture(scope="session", autouse=True)
def torch_jit_file(tmp_path_session):
    model = MockNet()
    torch.jit.script(model).save(tmp_path_session / "jit_model.pt")
    return tmp_path_session / "jit_model.pt"


@pytest.fixture
def wsi_with_annotations(wsi):
    """Fixture that provides a WSI with annotations for testing."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    import lazyslide as zs

    # Ensure tissues are segmented
    if "tissues" not in wsi.shapes:
        zs.pp.find_tissues(wsi)

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


def pytest_collection_modifyitems(config, items):
    if os.getenv("GITHUB_ACTIONS") == "true":
        skip_on_ci = pytest.mark.skip(reason="Skipped on GitHub CI")
        for item in items:
            if "skip_on_ci" in item.keywords:
                item.add_marker(skip_on_ci)
