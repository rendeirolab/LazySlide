import pytest

import lazyslide as zs

TIMM_MODEL = "mobilenetv3_small_050"


class TestFeatureExtraction:
    def test_load_model(self, wsi_small, torch_model_file):
        zs.tl.feature_extraction(wsi_small, model_path=torch_model_file)
        # Test feature aggregation
        zs.tl.feature_aggregation(wsi_small, feature_key="MockNet")

    def test_load_jit_model(self, wsi_small, torch_jit_file):
        zs.tl.feature_extraction(wsi_small, model_path=torch_jit_file)

    def test_timm_model(self, wsi_small):
        zs.tl.feature_extraction(wsi_small, model=TIMM_MODEL)


@pytest.mark.large_runner
class TestSlideEncoders:
    """Tests for all slide encoder models in lazyslide."""

    def test_madeleine_encoder(self, wsi_small):
        """Test the Madeleine slide encoder."""
        # Extract features with CONCH model
        zs.tl.feature_extraction(wsi_small, model="conch")

        # Test Madeleine slide encoder
        zs.tl.feature_aggregation(wsi_small, feature_key="conch", encoder="madeleine")
        assert "agg_slide" in wsi_small.tables["conch_tiles"].uns["agg_ops"]

    def test_titan_encoder(self, wsi_small):
        """Test the Titan slide encoder."""
        # Extract features with Titan model
        zs.tl.feature_extraction(wsi_small, model="titan")

        # Test Titan slide encoder
        zs.tl.feature_aggregation(wsi_small, feature_key="titan", encoder="titan")
        assert "agg_slide" in wsi_small.tables["titan_tiles"].uns["agg_ops"]

    def test_chief_encoder(self, wsi_small):
        """Test the CHIEF slide encoder."""
        # Extract features with CHIEF model
        zs.tl.feature_extraction(wsi_small, model="chief")

        # Test CHIEF slide encoder
        zs.tl.feature_aggregation(wsi_small, feature_key="chief", encoder="chief")
        assert "agg_slide" in wsi_small.tables["chief_tiles"].uns["agg_ops"]

    def test_prism_encoder(self, wsi_small):
        """Test the Prism slide encoder."""
        # Extract features with Virchow model
        zs.tl.feature_extraction(wsi_small, model="virchow")

        # Test Prism slide encoder
        zs.tl.feature_aggregation(wsi_small, feature_key="virchow", encoder="prism")
        assert "agg_slide" in wsi_small.tables["virchow_tiles"].uns["agg_ops"]

        # Check that latents were generated
        agg_info = wsi_small.tables["virchow_tiles"].uns["agg_ops"]["agg_slide"]
        assert "latents" in agg_info
