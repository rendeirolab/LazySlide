import numpy as np
import pandas as pd
from mock_models import MockPrismModel

import lazyslide as zs

TIMM_MODEL = "test_resnet"


class TestZeroShotClassification:
    def test_zero_shot_with_mock_prism(self, wsi):
        """Test zero-shot classification with mock Prism model."""
        # Prepare the WSI with necessary preprocessing
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 512)

        # Extract features using a lightweight timm model
        zs.tl.feature_extraction(wsi, model=TIMM_MODEL, load_kws=dict(pretrained=False))

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
