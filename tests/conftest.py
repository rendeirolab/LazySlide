import os

import pytest
import torch

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


# def pytest_configure(config):
#     """
#     Set a unique HF_HOME for each test session to avoid data races when using xdist.
#
#     This is particularly important when running tests in parallel on GitHub Actions
#     with matrix strategy across different Python versions and operating systems.
#     """
#     # Get the worker ID if running with pytest-xdist
#     worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
#
#     # Create a unique directory name using worker ID and process ID
#     unique_id = f"{worker_id}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
#
#     # Create a temporary directory for this test session
#     hf_home_dir = os.path.join(tempfile.gettempdir(), f"hf_home_{unique_id}")
#     os.makedirs(hf_home_dir, exist_ok=True)
#
#     # Set the HF_HOME environment variable
#     os.environ["HF_HOME"] = hf_home_dir
#
#     # Store the directory path for cleanup in pytest_unconfigure
#     config.hf_home_dir = hf_home_dir
#
#     # Log the HF_HOME directory for debugging
#     print(f"Setting HF_HOME to {hf_home_dir}")
#
#
# def pytest_unconfigure(config):
#     """
#     Clean up the temporary HF_HOME directory after tests are complete.
#     """
#     import shutil
#
#     # Check if we created a temporary directory in pytest_configure
#     if hasattr(config, "hf_home_dir") and os.path.exists(config.hf_home_dir):
#         try:
#             # Remove the temporary directory and all its contents
#             shutil.rmtree(config.hf_home_dir)
#             print(f"Cleaned up temporary HF_HOME directory: {config.hf_home_dir}")
#         except Exception as e:
#             print(f"Failed to clean up temporary HF_HOME directory: {e}")
