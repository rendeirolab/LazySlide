"""virtual_stain dispatches on the model's protocol, not its name.

Every model here is a fake built on the lazyslide-models base classes, so no
weights are downloaded. The slides are real test datasets.
"""

import numpy as np
import pytest

from lazyslide.tools import virtual_stain
from lazyslide.tools._virtual_staining import _rosie_postprocess, _tile_grid_index

from .mock_models import (
    MockMarkerMapModel,
    MockRosieModel,
    MockSemanticSegmentationModel,
    MockTileModel,
    MockVirtualStainModel,
)

RUN = {"batch_size": 8, "num_workers": 0, "pbar": False}


@pytest.fixture
def stained(request):
    """Remove whatever images a test adds, so the shared fixture stays clean."""
    wsi = request.getfixturevalue(request.param)
    before = set(wsi.images.keys())
    yield wsi
    for key in set(wsi.images.keys()) - before:
        del wsi.images[key]


def _covered(image):
    """Pixels that at least one tile wrote to."""
    return np.asarray(image).sum(axis=0) != 0


# ── Per-tile models ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("stained", ["wsi"], indirect=True)
def test_rosie_keeps_raw_values_by_default(stained):
    """ROSIE's display stretch is opt-in upstream (``--postprocess_image``),
    which recommends the raw values for quantitative analysis."""
    virtual_stain(stained, model=MockRosieModel(), **RUN)

    image = stained.images["rosie_prediction"]
    assert image.dims == ("c", "y", "x")
    assert list(image.c.values) == list(MockRosieModel.columns)
    assert image.dtype == np.float32
    values = np.asarray(image)[:, _covered(image)]
    assert values.min() >= 1 and values.max() < 11  # what the mock predicts
    assert "global" in image.attrs.get("transform", {})


@pytest.mark.parametrize("stained", ["wsi"], indirect=True)
def test_rosie_postprocess_stretches_into_uint8(stained):
    virtual_stain(stained, model=MockRosieModel(), postprocess=True, **RUN)

    image = stained.images["rosie_prediction"]
    assert list(image.c.values) == list(MockRosieModel.columns)
    assert image.dtype == np.uint8


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_postprocess_is_rosies_alone(stained):
    """Applied to another model it would replace the predicted values with a
    per-slide contrast stretch, so it is refused before any inference."""
    with pytest.raises(ValueError, match="ROSIE"):
        virtual_stain(stained, model=MockTileModel(), postprocess=True, **RUN)


def test_rosie_postprocess_background_is_the_90th_percentile():
    """As in ROSIE's evaluate.py, everything up to the 90th percentile of a
    channel is background, and the rest is stretched up to its 99.9th.

    The old stretch meant to start at the 1st percentile, but an inverted
    check reset that to 0 whenever the channel had any spread, so it only
    rescaled.
    """
    image = np.random.default_rng(0).random((50, 50, 1)).astype(np.float32) + 1
    rows, cols = np.indices((50, 50)).reshape(2, -1)

    out = _rosie_postprocess(image, rows, cols)

    assert out.dtype == np.uint8
    assert (out == 0).mean() >= 0.9


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_other_per_tile_models_keep_raw_values(stained):
    """Regression: ROSIE's contrast stretch used to apply to every tile model.

    A focus score or a class probability was clipped to percentiles, stretched
    into uint8 and median-blurred, so the stored image no longer held the
    values the model predicted.
    """
    model = MockTileModel()
    virtual_stain(stained, model=model, **RUN)

    image = stained.images["fake_tile_prediction"]
    assert list(image.c.values) == list(model.columns)
    assert image.dtype == np.float32
    covered = _covered(image)
    # One pixel per tile, holding exactly what the model predicted.
    assert covered.sum() == len(stained.shapes["tiles"])
    for channel, expected in enumerate(model.values):
        np.testing.assert_array_equal(np.asarray(image[channel])[covered], expected)


@pytest.mark.parametrize("fixture", ["wsi", "wsi_small"])
def test_every_tile_gets_its_own_pixel(fixture, request):
    """Regression: the old placement put several tiles on one pixel.

    It scaled by ``(H // stride) / H``, a little under ``1 / stride``, so
    rounding error built up across the slide. On ``wsi_small`` five of 31
    tiles landed on a pixel already taken, overwriting a neighbour.
    """
    wsi = request.getfixturevalue(fixture)
    spec = wsi.tile_spec("tiles")
    ds = wsi.ds.tile_images(tile_key="tiles")
    ys = np.array([ds[i]["y"] for i in range(len(ds))])
    xs = np.array([ds[i]["x"] for i in range(len(ds))])

    rows, cols = _tile_grid_index(ys, xs, spec)

    assert len(set(zip(rows.tolist(), cols.tolist()))) == len(ds)


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_per_tile_model_rejects_non_numeric_columns(stained):
    """A string column (e.g. a predicted class label) cannot become a pixel."""

    class Labeller(MockTileModel):
        columns = ("label",)

        def predict(self, image):
            return {"label": np.array(["tumour"] * image.shape[0])}

    with pytest.raises(TypeError, match="label"):
        virtual_stain(stained, model=Labeller(), **RUN)


# ── Dense marker maps ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_marker_map_is_stitched_without_a_second_activation(stained):
    """``predict`` returns final values, so the runner must not activate them.

    The old runner applied a sigmoid to GigaTIME output. With the model now
    applying its own, that would squash 0.25 to 0.562 and 0.75 to 0.679.
    """
    model = MockMarkerMapModel()
    virtual_stain(stained, model=model, **RUN)

    image = stained.images["fake_marker_map_prediction"]
    assert list(image.c.values) == list(model.channel_names)
    covered = _covered(image)
    assert covered.any()
    for channel, expected in enumerate(model.values):
        np.testing.assert_allclose(
            np.asarray(image[channel])[covered], expected, atol=1e-5
        )


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_any_marker_map_is_accepted_whatever_its_name(stained):
    """Regression: the old runner only knew "rosie" and "gigatime".

    ``gigatime-flash`` was registered but raised ``Model gigatime-flash not
    supported``, because the runner matched on names.
    """
    virtual_stain(stained, model=MockMarkerMapModel(name="gigatime-flash"), **RUN)
    assert "gigatime-flash_prediction" in stained.images


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_output_mpp_sets_the_resolution_of_the_stitched_image(stained):
    """A model predicting onto a coarser grid produces a smaller image."""
    tile_mpp = stained.tile_spec("tiles").mpp
    virtual_stain(stained, model=MockMarkerMapModel(name="native"), **RUN)
    virtual_stain(
        stained,
        model=MockMarkerMapModel(name="coarse", output_mpp=tile_mpp * 2),
        **RUN,
    )

    native = stained.images["native_prediction"]
    coarse = stained.images["coarse_prediction"]
    assert coarse.shape[1] == pytest.approx(native.shape[1] / 2, abs=1)
    assert coarse.shape[2] == pytest.approx(native.shape[2] / 2, abs=1)
    # Values survive the change of grid.
    covered = _covered(coarse)
    np.testing.assert_allclose(np.asarray(coarse[0])[covered], 0.25, atol=1e-5)


# ── Dense virtual stains ──────────────────────────────────────────────────────


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_each_stain_becomes_its_own_rgb_image(stained):
    model = MockVirtualStainModel()
    virtual_stain(stained, model=model, **RUN)

    # "HER2 IHC" has a space, which spatialdata rejects in element names.
    expected = {
        "fake_stain_prediction_PAS": model.colours[0],
        "fake_stain_prediction_HER2_IHC": model.colours[1],
    }
    assert set(expected) <= set(stained.images.keys())

    for key, colour in expected.items():
        image = stained.images[key]
        assert list(image.c.values) == ["r", "g", "b"]
        covered = _covered(image)
        for channel, expected in enumerate(colour):
            np.testing.assert_allclose(
                np.asarray(image[channel])[covered], expected, atol=1e-5
            )


# ── Anything else is refused ──────────────────────────────────────────────────


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_model_with_neither_protocol_is_rejected(stained):
    with pytest.raises(TypeError, match="virtual_stain"):
        virtual_stain(stained, model=MockSemanticSegmentationModel(), **RUN)


def test_unknown_model_name_is_rejected(wsi):
    with pytest.raises(KeyError):
        virtual_stain(wsi, model="unsupported_model", **RUN)


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_dense_model_that_is_neither_kind_is_rejected(stained):
    """A dense model must be a marker map or a virtual stain to be stored.

    Without an explicit check, one that is neither (for example a marker map
    that forgot ``channel_names``) got past the guard and failed later with an
    ``AttributeError`` instead of saying what was wrong.
    """
    from lazyslide_models.base import DensePredictionModel

    class Unnamed(DensePredictionModel):
        def __init__(self):
            self.model = None

        def get_transform(self):
            return None

        def predict(self, image):
            raise AssertionError("must be rejected before predicting")

    with pytest.raises(TypeError, match="marker map or a virtual stain"):
        virtual_stain(stained, model=Unnamed(), **RUN)


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_stains_that_collide_after_renaming_are_rejected(stained):
    """spatialdata-safe names are not unique: "HER2 IHC" and "HER2/IHC" both
    become ``HER2_IHC``, so the second image would silently replace the first."""

    class Colliding(MockVirtualStainModel):
        stains = ("HER2 IHC", "HER2/IHC")

    before = set(stained.images.keys())
    with pytest.raises(ValueError, match="HER2_IHC"):
        virtual_stain(stained, model=Colliding(), **RUN)
    # Nothing is half-written.
    assert set(stained.images.keys()) == before


@pytest.mark.parametrize("stained", ["wsi_small"], indirect=True)
def test_dense_output_survives_a_backing_file_that_cannot_be_deleted(
    stained, monkeypatch, tmp_path
):
    """Reproduces the Windows failure on any OS.

    A dense result is a memmap in a temporary directory, and the stored image
    still maps it when the directory is cleaned up. POSIX lets a mapped file be
    deleted; Windows refuses with ``PermissionError``, which used to crash
    ``virtual_stain`` for every marker map and virtual stain there.
    """
    import os
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # leftovers land here
    real_unlink = os.unlink

    def unlink_like_windows(path, *args, **kwargs):
        if os.path.basename(os.fspath(path)) == "image.npy":
            raise PermissionError(32, "being used by another process", path)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink_like_windows)

    model = MockMarkerMapModel()
    virtual_stain(stained, model=model, **RUN)

    image = stained.images["fake_marker_map_prediction"]
    covered = _covered(image)
    np.testing.assert_allclose(np.asarray(image[0])[covered], 0.25, atol=1e-5)
