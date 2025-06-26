import pytest

import lazyslide as zs
from lazyslide._const import Key
from lazyslide.models import MODEL_REGISTRY

TIMM_MODEL = "mobilenetv3_small_050"
VISION_MODELS = [
    model_name
    for model_name, model_info in MODEL_REGISTRY.items()
    if model_info.model_type == "vision"
]


class TestFeatureExtraction:
    def test_load_model(self, wsi, torch_model_file):
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 512)
        zs.tl.feature_extraction(wsi, model_path=torch_model_file)
        # Test feature aggregation
        zs.tl.feature_aggregation(wsi, feature_key="MockNet")

    def test_load_jit_model(self, wsi, torch_jit_file):
        zs.tl.feature_extraction(wsi, model_path=torch_jit_file)

    @pytest.mark.skip_on_ci
    def test_timm_model(self, wsi):
        zs.tl.feature_extraction(wsi, model=TIMM_MODEL)

    @pytest.mark.parametrize("model_name", VISION_MODELS)
    def test_vision_model(self, wsi, model_name):
        """Test feature extraction with all vision models."""
        # Prepare the WSI
        if "tiles" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
            zs.pp.tile_tissues(wsi, 512)

        # Extract features
        zs.tl.feature_extraction(wsi, model=model_name)

        # Check that features were extracted
        feature_key = Key.feature(model_name, "tiles")
        assert feature_key in wsi.tables
        assert wsi.tables[feature_key].X.shape[0] == len(wsi.shapes["tiles"])


@pytest.mark.skip_on_ci
class TestSlideEncoders:
    """Tests for all slide encoder models in lazyslide."""

    def test_madeleine_encoder(self, wsi):
        """Test the Madeleine slide encoder."""
        # Prepare the WSI
        if "tiles" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
            zs.pp.tile_tissues(wsi, 512)

        # Extract features with CONCH model
        zs.tl.feature_extraction(wsi, model="conch")

        # Test Madeleine slide encoder
        zs.tl.feature_aggregation(wsi, feature_key="conch", encoder="madeleine")
        assert "agg_slide" in wsi.tables["conch_tiles"].uns["agg_ops"]

    def test_titan_encoder(self, wsi):
        """Test the Titan slide encoder."""
        # Prepare the WSI
        if "tiles" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
            zs.pp.tile_tissues(wsi, 512)

        # Extract features with Titan model
        zs.tl.feature_extraction(wsi, model="titan")

        # Test Titan slide encoder
        zs.tl.feature_aggregation(wsi, feature_key="titan", encoder="titan")
        assert "agg_slide" in wsi.tables["titan_tiles"].uns["agg_ops"]

    def test_chief_encoder(self, wsi):
        """Test the CHIEF slide encoder."""
        # Prepare the WSI
        if "tiles" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
            zs.pp.tile_tissues(wsi, 512)

        # Extract features with CHIEF model
        zs.tl.feature_extraction(wsi, model="chief")

        # Test CHIEF slide encoder
        zs.tl.feature_aggregation(wsi, feature_key="chief", encoder="chief")
        assert "agg_slide" in wsi.tables["chief_tiles"].uns["agg_ops"]

    def test_prism_encoder(self, wsi):
        """Test the Prism slide encoder."""
        # Prepare the WSI
        if "tiles" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
            zs.pp.tile_tissues(wsi, 512)

        # Extract features with Virchow model
        zs.tl.feature_extraction(wsi, model="virchow")

        # Test Prism slide encoder
        zs.tl.feature_aggregation(wsi, feature_key="virchow", encoder="prism")
        assert "agg_slide" in wsi.tables["virchow_tiles"].uns["agg_ops"]

        # Check that latents were generated
        agg_info = wsi.tables["virchow_tiles"].uns["agg_ops"]["agg_slide"]
        assert "latents" in agg_info
