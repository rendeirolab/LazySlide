import numpy as np
import pandas as pd
import pytest

import lazyslide as zs


class TestZeroShotClassification:
    @pytest.mark.large_runner
    def test_zero_shot_with_real_prism(self, wsi):
        """Test zero-shot classification with actual Prism model.

        This test is marked to be skipped on CI to avoid downloading large models.
        Run it locally if you want to test with the actual Prism model.
        """
        # Skip this test if running on CI
        pytest.importorskip("transformers")

        # Prepare the WSI with necessary preprocessing
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 512)

        # Extract features using a simple model
        zs.tl.feature_extraction(wsi, model="virchow")

        # Aggregate features
        zs.tl.feature_aggregation(wsi, feature_key="virchow", encoder="prism")

        # Define prompts for zero-shot classification
        prompts = [["normal tissue"], ["abnormal tissue"], ["inflammation"]]

        # Perform zero-shot classification
        results = zs.tl.zero_shot_score(
            wsi, prompts, feature_key="virchow_tiles", model="prism"
        )

        # Verify the results
        assert isinstance(results, pd.DataFrame)
        assert results.shape[1] == len(prompts)  # Number of classes
        assert list(results.columns) == [
            "normal tissue",
            "abnormal tissue",
            "inflammation",
        ]

        # Check that probabilities sum to approximately 1
        assert np.isclose(results.sum(axis=1).values[0], 1.0)
