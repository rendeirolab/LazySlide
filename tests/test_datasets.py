import os

import pytest

import lazyslide as zs

# These tests load datasets directly (no fixtures), so they must honor the same
# skip flag conftest uses for fork PRs / runs without an HF token.
pytestmark = pytest.mark.skipif(
    os.environ.get("LAZYSLIDE_SKIP_DATASET_TESTS") == "1",
    reason="Dataset tests skipped (no HF token, e.g. fork PR)",
)


def test_load_sample():
    wsi = zs.datasets.sample()
    assert wsi is not None
    assert "tissues" in wsi.shapes


def test_load_gtex_artery():
    wsi = zs.datasets.gtex_artery()
    assert wsi is not None
    assert "tissues" in wsi.shapes
