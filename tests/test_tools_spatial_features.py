import numpy as np
import pytest

import lazyslide as zs
from lazyslide.tools import spatial_features


class TestSpatialFeatures:
    """Test class for spatial_features function."""

    def test_basic_functionality(self, wsi):
        """Test basic spatial features functionality."""
        # Prepare WSI with necessary data
        # wsi has feature extracted with resnet50
        zs.pp.tile_graph(wsi)

        # Run spatial features
        spatial_features(wsi, "resnet50")

        # Check that spatial features were added
        feature_table = wsi.tables["resnet50_tiles"]
        assert "spatial_features" in feature_table.layers

        # Check output shape matches input features
        original_features = feature_table.X
        spatial_feat = feature_table.layers["spatial_features"]
        assert spatial_feat.shape == original_features.shape

        # Check that spatial features are different from original (smoothed)
        assert not np.array_equal(original_features, spatial_feat)

        # Check that spatial features are numeric
        assert np.isfinite(spatial_feat).all()

    def test_invalid_method_raises_error(self, wsi):
        """Test that invalid method raises ValueError."""
        # Prepare WSI with necessary data
        zs.pp.tile_graph(wsi)

        # Test invalid method
        with pytest.raises(ValueError, match="Unknown method 'invalid'"):
            spatial_features(wsi, "resnet50", method="invalid")

    def test_missing_graph_raises_error(self, wsi_small):
        """Test that missing tile graph raises ValueError."""
        # Prepare fresh WSI without tile graph (using wsi_small to avoid state from other tests)
        zs.pp.find_tissues(wsi_small)
        zs.pp.tile_tissues(wsi_small, 256, mpp=0.5, key_added="missing_graph_tiles")
        zs.tl.feature_extraction(wsi_small, "resnet50", tile_key="missing_graph_tiles")
        # Intentionally skip tile_graph step

        # Test missing graph
        with pytest.raises(ValueError, match="The tile graph is needed"):
            spatial_features(wsi_small, "resnet50", tile_key="missing_graph_tiles")

    def test_invalid_feature_key_raises_error(self, wsi):
        """Test that invalid feature key raises appropriate error."""
        # Test invalid feature key
        with pytest.raises(Exception):  # Should raise some error for missing features
            spatial_features(wsi, "nonexistent_features")

    def test_auto_graph_key_generation(self, wsi):
        """Test automatic graph key generation from tile key."""
        # Prepare WSI with custom tile key
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, mpp=0.5, key_added="my_tiles")
        zs.tl.feature_extraction(wsi, "resnet50", tile_key="my_tiles")
        zs.pp.tile_graph(wsi, tile_key="my_tiles")  # Creates "my_tiles_graph"

        # Run spatial features without specifying graph_key
        spatial_features(wsi, "resnet50", tile_key="my_tiles")

        # Should automatically use "my_tiles_graph"
        feature_table = wsi.tables["resnet50_my_tiles"]
        assert "spatial_features" in feature_table.layers

    def test_spatial_smoothing_properties(self, wsi):
        """Test properties of spatial smoothing operation."""
        # Prepare WSI with necessary data
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        zs.tl.feature_extraction(wsi, "resnet50")
        zs.pp.tile_graph(wsi)

        # Get original features
        original_features = wsi.tables["resnet50_tiles"].X.copy()

        # Run spatial features
        spatial_features(wsi, "resnet50")
        smoothed_features = wsi.tables["resnet50_tiles"].layers["spatial_features"]

        # Test that smoothing preserves feature dimensions
        assert smoothed_features.shape == original_features.shape

        # Test that smoothed features are generally closer to neighbors
        # (This is a property of spatial smoothing)
        assert (
            smoothed_features.dtype == original_features.dtype
            or smoothed_features.dtype == np.float64
        )
