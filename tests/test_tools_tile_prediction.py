import numpy as np
import pytest

import lazyslide as zs
from lazyslide.tools import tile_prediction


class TestTilePrediction:
    """Test class for tile_prediction function."""

    def test_basic_functionality_cv_features(self, wsi_small):
        """Test basic tile prediction functionality with CV features model."""
        # Prepare WSI with tiles
        # Run tile prediction with a CV feature model
        tile_prediction(wsi_small, "split_rgb")

        # Check that predictions were added to WSI
        assert "tiles" in wsi_small.shapes
        tiles_gdf = wsi_small.shapes["tiles"]

        # Verify that features were added (CV features add columns directly)
        assert (
            len(tiles_gdf.columns) > 3
        )  # Should have more than just geometry and basic columns

        # Check that all tiles have predictions
        assert len(tiles_gdf) > 0

    def test_string_model_name(self, wsi_small):
        """Test tile prediction with string model name."""
        # Prepare WSI with tiles
        # Test with different CV feature models
        for model_name in ["brightness", "contrast", "sobel"]:
            tile_prediction(wsi_small, model_name)

            # Check that tiles exist and have data
            tiles_gdf = wsi_small.shapes["tiles"]
            assert len(tiles_gdf) > 0
            assert len(tiles_gdf.columns) > 3

    def test_custom_tile_key(self, wsi_small):
        """Test tile prediction with custom tile key."""
        # Prepare WSI with custom tile key
        zs.pp.tile_tissues(wsi_small, 256, mpp=0.5, key_added="custom_tiles")

        # Run tile prediction with custom tile key
        tile_prediction(wsi_small, "split_rgb", tile_key="custom_tiles")

        # Check that predictions were added to custom tiles
        assert "custom_tiles" in wsi_small.shapes
        tiles_gdf = wsi_small.shapes["custom_tiles"]
        assert len(tiles_gdf) > 0
        assert len(tiles_gdf.columns) > 3

    def test_custom_transform(self, wsi_small):
        """Test tile prediction with custom transform."""

        # Prepare WSI with tiles
        # Define a simple custom transform
        def custom_transform(image):
            return image

        # Run tile prediction with custom transform
        tile_prediction(wsi_small, "split_rgb", transform=custom_transform)

        # Check that predictions were added
        tiles_gdf = wsi_small.shapes["tiles"]
        assert len(tiles_gdf) > 0

    def test_invalid_model_string_raises_error(self, wsi_small):
        """Test that invalid model string raises KeyError."""
        # Test invalid model name
        with pytest.raises(KeyError, match="Cannot find model"):
            tile_prediction(wsi_small, "nonexistent_model")

    def test_spider_model_without_variant_raises_error(self, wsi_small):
        """Test that 'spider' model without variant raises ValueError."""
        # Test spider model without variant
        with pytest.raises(
            ValueError, match="For spider model, please specify the variants"
        ):
            tile_prediction(wsi_small, "spider")

    def test_invalid_model_type_raises_error(self, wsi_small):
        """Test that invalid model type raises appropriate error."""
        # Test with invalid model type
        with pytest.raises(Exception):  # Should raise some error
            tile_prediction(wsi_small, 123)  # Invalid type

    def test_minimal_parameters(self, wsi_small):
        """Test tile prediction with minimal required parameters."""
        # Run with minimal parameters (just wsi and model)
        tile_prediction(wsi_small, "brightness")

        # Check that it worked
        tiles_gdf = wsi_small.shapes["tiles"]
        assert len(tiles_gdf) > 0
        assert len(tiles_gdf.columns) > 3

    def test_predictions_added_to_wsi(self, wsi_small):
        """Test that predictions are properly added to WSIData object."""
        # Get initial column count
        initial_columns = len(wsi_small.shapes["tiles"].columns)

        # Run tile prediction
        tile_prediction(wsi_small, "brightness")

        # Check that new columns were added
        final_columns = len(wsi_small.shapes["tiles"].columns)
        assert final_columns > initial_columns

        # Check that all values are finite (no NaN/inf)
        tiles_gdf = wsi_small.shapes["tiles"]
        for col in tiles_gdf.columns:
            if tiles_gdf[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                assert np.isfinite(tiles_gdf[col]).all(), (
                    f"Column {col} contains non-finite values"
                )

    def test_multiple_cv_features(self, wsi_small):
        """Test multiple CV feature models on the same WSI."""
        # Prepare WSI with tiles
        zs.pp.find_tissues(wsi_small)
        zs.pp.tile_tissues(wsi_small, 256, mpp=0.5)

        # Run multiple CV feature extractions
        models_to_test = ["brightness", "contrast", "saturation"]
        for model in models_to_test:
            tile_prediction(wsi_small, model)

        # Check that all features were added
        tiles_gdf = wsi_small.shapes["tiles"]
        assert len(tiles_gdf) > 0
        # Should have accumulated features from all models (3 base columns + 3 features = 6)
        assert len(tiles_gdf.columns) >= 6  # Base columns + multiple features
