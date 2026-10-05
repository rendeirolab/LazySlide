import os
import re
import shutil
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pytest
import tifffile
from shapely import box
from wsidata import open_wsi
from wsidata.io import add_shapes

import lazyslide as zs
from lazyslide.datasets import _sample

# These tests load datasets directly (no fixtures), so they must honor the same
# skip flag conftest uses for fork PRs / runs without an HF token.
needs_hf = pytest.mark.skipif(
    os.environ.get("LAZYSLIDE_SKIP_DATASET_TESTS") == "1",
    reason="Dataset tests skipped (no HF token, e.g. fork PR)",
)


@needs_hf
def test_load_sample():
    wsi = zs.datasets.sample()
    assert wsi is not None
    assert "tissues" in wsi.shapes


@needs_hf
def test_load_gtex_artery():
    wsi = zs.datasets.gtex_artery()
    assert wsi is not None
    assert "tissues" in wsi.shapes


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A snapshot folder of the Hugging Face cache: a slide and its zipped store."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    slide = snapshot / "fake.tiff"
    tifffile.imwrite(
        slide, np.zeros((512, 512, 3), np.uint8), tile=(256, 256), photometric="rgb"
    )
    build = tmp_path / "build"
    wsi = open_wsi(slide, store=build / "fake.zarr")
    add_shapes(wsi, "tissues", gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)]))
    wsi.write()
    shutil.make_archive(
        str(snapshot / "fake.zarr"), "zip", root_dir=build, base_dir="fake.zarr"
    )

    monkeypatch.setenv("LAZYSLIDE_DATASET_REVISION", "main")
    monkeypatch.setattr(
        _sample,
        "hf_hub_download",
        lambda repo_id, filename, **kwargs: str(snapshot / filename),
    )
    return snapshot


def test_open_wsi_does_not_attach_dataset_store(fake_hub):
    wsi = _sample._load_dataset("fake.tiff", "fake.zarr.zip")
    assert "tissues" in wsi.shapes
    # A write() or rmtree of the store open_wsi attaches would change the dataset
    assert "tissues" not in open_wsi(fake_hub / "fake.tiff").shapes


def test_dataset_store_extracted_next_to_slide_is_kept(fake_hub):
    # Where earlier versions extracted the store, written to since by the user
    old_store = fake_hub / "fake.zarr"
    old_store.mkdir()
    (old_store / "user-data").touch()

    with pytest.warns(UserWarning, match=re.escape(str(old_store))):
        wsi = _sample._load_dataset("fake.tiff", "fake.zarr.zip")

    assert "tissues" in wsi.shapes
    assert (old_store / "user-data").exists()


def test_interrupted_extraction_leaves_no_partial_store(fake_hub, monkeypatch):
    def extract_half_then_fail(self, path=None, members=None, pwd=None):
        names = self.namelist()
        for name in names[: len(names) // 2]:
            self.extract(name, path, pwd)
        raise OSError("No space left on device")

    with monkeypatch.context() as m:
        m.setattr(ZipFile, "extractall", extract_half_then_fail)
        with pytest.raises(OSError, match="No space left"):
            _sample._load_dataset("fake.tiff", "fake.zarr.zip")

    wsi = _sample._load_dataset("fake.tiff", "fake.zarr.zip")
    assert "tissues" in wsi.shapes


def test_dataset_store_moved_in_by_another_load_meanwhile(fake_hub, monkeypatch):
    extractall = ZipFile.extractall
    other_load = []

    def another_load_finishes_first(self, *args, **kwargs):
        monkeypatch.setattr(ZipFile, "extractall", extractall)
        other_load.append(_sample._load_dataset("fake.tiff", "fake.zarr.zip"))
        extractall(self, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "extractall", another_load_finishes_first)
    wsi = _sample._load_dataset("fake.tiff", "fake.zarr.zip")

    assert "tissues" in other_load[0].shapes
    assert "tissues" in wsi.shapes
