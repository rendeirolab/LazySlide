import numpy as np
import pandas as pd

import lazyslide as zs

from .mock_models import MockPrismModel

TIMM_MODEL = "test_resnet"


class TestZeroShotClassification:
    def test_zero_shot_with_mock_prism(self, wsi):
        """Test zero-shot classification with mock Prism model."""
        # Prepare the WSI with necessary preprocessing
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 512)

        # Extract features using a lightweight timm model
        zs.tl.feature_extraction(wsi, model=TIMM_MODEL, load_kws={"pretrained": False})

        # Aggregate features using mock prism
        mock_prism = MockPrismModel()
        # feature_aggregation uses MODEL_REGISTRY for encoder string,
        # so use "mean" for aggregation and then call zero_shot_score with model instance
        zs.tl.feature_aggregation(wsi, feature_key=TIMM_MODEL, encoder="mean")

        # Define prompts for zero-shot classification
        prompts = [["normal tissue"], ["abnormal tissue"], ["inflammation"]]

        # Perform zero-shot classification with mock model instance
        results = zs.tl.zero_shot_score(
            wsi,
            prompts,
            feature_key=f"{TIMM_MODEL}_tiles",
            model=mock_prism,
        )

        # Verify the results
        assert isinstance(results, pd.DataFrame)
        assert results.shape[1] == len(prompts)
        assert list(results.columns) == [
            "normal tissue",
            "abnormal tissue",
            "inflammation",
        ]

        # Check that probabilities sum to approximately 1
        assert np.isclose(results.sum(axis=1).values[0], 1.0)

    def test_zero_shot_accepts_flat_prompts(self, wsi):
        """A plain list of strings is one class per string.

        This is the form used in the ``zero_shot_score`` docstring example, and
        the reason its ``prompts`` annotation is ``list[str | list[str]]``.
        """
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 512)
        zs.tl.feature_extraction(wsi, model=TIMM_MODEL, load_kws={"pretrained": False})
        zs.tl.feature_aggregation(wsi, feature_key=TIMM_MODEL, encoder="mean")

        results = zs.tl.zero_shot_score(
            wsi,
            ["lung cancer", "normal lung"],
            feature_key=f"{TIMM_MODEL}_tiles",
            model=MockPrismModel(),
        )

        assert list(results.columns) == ["lung cancer", "normal lung"]
        assert np.isclose(results.sum(axis=1).values[0], 1.0)


def test_sam_probabilities_are_thresholded_at_half():
    """SAM's ``segment`` returns sigmoid probabilities, not a mask.

    Cast with ``.astype(bool)``, every probability above zero became
    foreground. SAM's own mask threshold is logit 0, probability 0.5.
    """
    import torch
    from lazyslide_models.base import SegmentationOutput

    from lazyslide.segmentation._zero_shot import _segment_with_model

    prob = torch.full((1, 1, 4, 6), 0.1)
    prob[..., :3] = 0.9

    class Sam:
        def segment(self, image, **kwargs):
            return SegmentationOutput(probability_map=prob)

    image = np.zeros((4, 6, 3), dtype=np.uint8)
    mask = _segment_with_model(Sam(), image, None, [[1, 1]], [], [[0, 0, 3, 4]])

    expected = np.zeros((4, 6), dtype=bool)
    expected[:, :3] = True
    np.testing.assert_array_equal(mask, expected)
