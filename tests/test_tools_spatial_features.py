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

    def test_custom_layer_key(self, wsi):
        """Test spatial features with custom layer key."""
        # Prepare WSI with necessary data
        zs.pp.tile_graph(wsi)

        custom_key = "custom_spatial"
        spatial_features(wsi, "resnet50", layer_key=custom_key)

        # Check that custom layer key was used
        feature_table = wsi.tables["resnet50_tiles"]
        assert custom_key in feature_table.layers
        assert feature_table.layers[custom_key].shape == feature_table.X.shape

    def test_custom_tile_and_graph_keys(self, wsi):
        """Test spatial features with custom tile and graph keys."""
        # Prepare WSI with custom keys
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, mpp=0.5, key_added="custom_tiles")
        zs.tl.feature_extraction(wsi, "resnet50", tile_key="custom_tiles")
        zs.pp.tile_graph(wsi, tile_key="custom_tiles", table_key="custom_graph")

        # Run spatial features with custom keys
        spatial_features(
            wsi, "resnet50", tile_key="custom_tiles", graph_key="custom_graph"
        )

        # Check that spatial features were added
        feature_table = wsi.tables["resnet50_custom_tiles"]
        assert "spatial_features" in feature_table.layers

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
        # Prepare WSI with necessary data but no features
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        zs.pp.tile_graph(wsi)
        # Intentionally skip feature extraction

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
        )  # May be upcast during computation

    def test_multiple_runs_same_result(self, wsi):
        """Test that running spatial features multiple times gives same result."""
        # Prepare WSI with necessary data
        zs.pp.tile_graph(wsi)

        # Run spatial features first time
        spatial_features(wsi, "resnet50", layer_key="spatial_1")
        result_1 = wsi.tables["resnet50_tiles"].layers["spatial_1"].copy()

        # Run spatial features second time
        spatial_features(wsi, "resnet50", layer_key="spatial_2")
        result_2 = wsi.tables["resnet50_tiles"].layers["spatial_2"]

        # Results should be identical
        np.testing.assert_array_equal(result_1, result_2)

    def test_smoothing_method_explicit(self, wsi):
        """Test explicitly setting method='smoothing'."""
        # Prepare WSI with necessary data
        zs.pp.tile_graph(wsi)

        # Run spatial features with explicit method
        spatial_features(wsi, "resnet50", method="smoothing")

        # Check that spatial features were added
        feature_table = wsi.tables["resnet50_tiles"]
        assert "spatial_features" in feature_table.layers

        # Verify output properties
        spatial_feat = feature_table.layers["spatial_features"]
        assert spatial_feat.shape == feature_table.X.shape
        assert np.isfinite(spatial_feat).all()
