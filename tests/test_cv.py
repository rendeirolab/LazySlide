import pytest
import numpy as np


np.random.seed(42)

H, W = 100, 100
N_CLASS = 5
binary_mask = np.random.randint(0, 2, (H, W), dtype=np.uint8)
multilabel_mask = np.random.randint(0, N_CLASS, (H, W), dtype=np.uint8)
multiclass_mask = np.random.randint(
    0,
    2,
    (
        N_CLASS,
        H,
        W,
    ),
    dtype=np.uint8,
)


class TestMask:
    @pytest.mark.parametrize("mask", [binary_mask, multilabel_mask, multiclass_mask])
    def test_mask_to_polygon(self, mask):
        from lazyslide.cv.mask import Mask

        mask = Mask.from_array(mask)
        mask.to_polygons()
