from inspect import signature

import cv2
import numpy as np
import pytest

from lazyslide.cv.transform import (
    Compose,
    EntropyThreshold,
    GaussianBlur,
    MedianBlur,
    MorphClose,
    mods,
)
from lazyslide.cv.transform.compose import Mask2Polygon
from lazyslide.cv.transform.mods import Transform
from lazyslide.preprocess._tissue import _tissue_mask, find_tissues


class TestEntropyThreshold:
    """Tests for the EntropyThreshold transform."""

    def test_default_params(self):
        """Test that default parameters are stored correctly."""
        t = EntropyThreshold()
        assert t.params["disk_radius"] == 4
        assert t.params["relaxed_threshold"] is True
        assert t.params["invert_check"] is True

    def test_custom_params_stored(self):
        """Test that custom parameters are stored correctly."""
        t = EntropyThreshold(disk_radius=6, relaxed_threshold=False, invert_check=False)
        assert t.params["disk_radius"] == 6
        assert t.params["relaxed_threshold"] is False
        assert t.params["invert_check"] is False

    def test_output_shape_and_dtype(self):
        """Test apply returns 2D uint8 mask matching H x W."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
        t = EntropyThreshold()
        mask = t.apply(image)

        assert mask.ndim == 2
        assert mask.shape == (64, 80)
        assert mask.dtype == np.uint8

    def test_output_values_are_0_or_255(self):
        """Test that mask values are in {0, 255}."""
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        t = EntropyThreshold()
        mask = t.apply(image)

        unique_vals = set(np.unique(mask).tolist())
        assert unique_vals.issubset({0, 255})

    def test_call_interface(self):
        """Test __call__ wraps apply and returns uint8."""
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        t = EntropyThreshold()
        mask = t(image)

        assert mask.shape == (48, 48)
        assert mask.dtype == np.uint8

    @pytest.mark.parametrize("disk_radius", [2, 4, 6])
    def test_disk_radius_param(self, disk_radius):
        """Test apply runs across disk_radius values."""
        rng = np.random.default_rng(disk_radius)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        t = EntropyThreshold(disk_radius=disk_radius)
        mask = t.apply(image)

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert t.params["disk_radius"] == disk_radius

    @pytest.mark.parametrize("relaxed_threshold", [True, False])
    def test_relaxed_threshold_param(self, relaxed_threshold):
        """Test apply runs with relaxed_threshold True and False."""
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        t = EntropyThreshold(relaxed_threshold=relaxed_threshold)
        mask = t.apply(image)

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert t.params["relaxed_threshold"] is relaxed_threshold

    @pytest.mark.parametrize("invert_check", [True, False])
    def test_invert_check_param(self, invert_check):
        """Test apply runs with invert_check True and False."""
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        t = EntropyThreshold(invert_check=invert_check)
        mask = t.apply(image)

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert t.params["invert_check"] is invert_check

    def test_uniform_white_image(self):
        """Test that a uniform white image produces a valid mask."""
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        t = EntropyThreshold()
        mask = t.apply(image)

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()).issubset({0, 255})

    def test_invert_check_flips_dominant_foreground(self):
        """Test invert_check inverts when border foreground exceeds 95%.

        Constructs an image where the border is high-frequency noise (high local
        entropy) and the center is a uniform low-entropy patch. After
        ``rgb2hed`` + rank entropy + Otsu, the border ends up as foreground and
        the center as background, so >95% of border pixels are foreground.
        ``invert_check=True`` should flip this; ``invert_check=False`` should
        leave it as-is.
        """
        rng = np.random.default_rng(0)
        H, W = 128, 128
        border = 16
        image = np.full((H, W, 3), 230, dtype=np.uint8)
        noise = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        border_mask = np.zeros((H, W), dtype=bool)
        border_mask[:border, :] = True
        border_mask[-border:, :] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True
        image[border_mask] = noise[border_mask]

        mask_no_invert = EntropyThreshold(invert_check=False).apply(image)

        # Sanity-check the construction: border foreground fraction > 0.95.
        bool_mask = mask_no_invert > 0
        top = bool_mask[0, :]
        right = bool_mask[:, -1]
        bottom = bool_mask[-1, :]
        left = bool_mask[:, 0]
        edge = np.concatenate([top, right[1:-1], bottom, left[1:-1]])
        assert edge.mean() > 0.95

        mask_invert = EntropyThreshold(invert_check=True).apply(image)

        # invert_check=True should bitwise-invert the mask.
        assert np.array_equal(mask_invert, 255 - mask_no_invert)

    def test_disk_radius_invalid_raises(self):
        """Test EntropyThreshold raises ValueError for disk_radius < 1."""
        with pytest.raises(ValueError):
            EntropyThreshold(disk_radius=0)
        with pytest.raises(ValueError):
            EntropyThreshold(disk_radius=-1)


class TestTransformParamsIsolation:
    """Regression tests: `params` must be per-instance, not a shared class dict."""

    def test_params_are_independent_between_instances(self):
        """Constructing a second transform must not overwrite the first's params."""
        first = MedianBlur(kernel_size=3)
        second = GaussianBlur(kernel_size=7, sigma=2)

        assert first.params == {"kernel_size": 3}
        assert second.params == {"kernel_size": 7, "sigma": 2}
        assert first.params is not second.params
        assert repr(first) == "MedianBlur(kernel_size=3)"

    def test_class_attribute_is_never_mutated(self):
        """The class-level default stays empty, so Compose reports no foreign params."""
        MorphClose(kernel_size=9, n_iterations=2)

        assert Transform.params == {}
        assert MedianBlur.params == {}
        assert Compose([]).params == {}
        assert repr(Compose([])) == "Compose()"

    def test_tissue_mask_blurs_with_blur_ksize(self, monkeypatch):
        """_tissue_mask must blur with blur_ksize, not the later MorphClose ksize."""
        seen = []
        real_median_blur = cv2.medianBlur

        def spy(src, ksize):
            seen.append(ksize)
            return real_median_blur(src, ksize)

        monkeypatch.setattr(mods.cv2, "medianBlur", spy)

        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        _tissue_mask(
            image,
            to_hsv=False,
            filter_artifacts=True,
            blur_ksize=17,
            threshold=7,
            morph_ksize=7,
            morph_n_iter=1,
        )

        assert seen == [17]

    def test_default_blur_ksize_preserves_legacy_output(self, monkeypatch):
        """The default blur kernel must stay 7 - the kernel the shared-params
        bug actually used - so default tissue masks are unchanged by the fix."""
        seen = []
        real_median_blur = cv2.medianBlur

        def spy(src, ksize):
            seen.append(ksize)
            return real_median_blur(src, ksize)

        monkeypatch.setattr(mods.cv2, "medianBlur", spy)

        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        _tissue_mask(image, to_hsv=False)

        # Pre-fix, MorphClose(kernel_size=morph_ksize) clobbered the shared
        # dict and the blur ran with morph_ksize. Both default to 7 now, so
        # the mask a default call produces is byte-identical to the old one.
        assert seen == [7]
        assert signature(_tissue_mask).parameters["blur_ksize"].default == 7
        assert signature(find_tissues).parameters["blur_ksize"].default == 7

    def test_find_tissues_blurs_with_blur_ksize(self, wsi, monkeypatch):
        """End-to-end: find_tissues(blur_ksize=X, morph_ksize=Y) blurs with X."""
        import lazyslide as zs

        seen = []
        real_median_blur = cv2.medianBlur

        def spy(src, ksize):
            seen.append(ksize)
            return real_median_blur(src, ksize)

        monkeypatch.setattr(mods.cv2, "medianBlur", spy)

        zs.pp.find_tissues(
            wsi, blur_ksize=21, morph_ksize=7, key_added="tissue_blur_ksize"
        )

        assert seen
        assert set(seen) == {21}


class _RaisingTransform:
    """Stand-in for a pipeline stage that fails."""

    def apply(self, mask):
        raise RuntimeError("stage failed")


class _RecordingTransform:
    """Stand-in for a pipeline stage that records whether it ran."""

    def __init__(self):
        self.called = False

    def apply(self, mask):
        self.called = True
        return mask


class TestMask2PolygonPipelineFailure:
    """Tests that a failing pipeline stage is not swallowed."""

    def test_middle_transform_error_propagates(self):
        """A raising stage propagates instead of continuing with a stale mask."""
        t = Mask2Polygon()
        downstream = _RecordingTransform()
        t.pipeline = [_RecordingTransform(), _RaisingTransform(), downstream]

        with pytest.raises(RuntimeError, match="stage failed"):
            t.apply(np.zeros((16, 16), dtype=np.uint8))

        assert not downstream.called
