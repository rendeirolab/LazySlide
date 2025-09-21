import json
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

import lazyslide as zs


class TestLoadAnnotations:
    """Tests for the io.load_annotations function."""

    @pytest.fixture
    def sample_geodataframe(self):
        """Create a sample GeoDataFrame for testing."""
        # Create larger polygons to handle min_area filtering
        polygons = [
            Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]),  # Area: 40,000
            Polygon([(300, 300), (600, 300), (600, 600), (300, 600)]),  # Area: 90,000
        ]
        gdf = gpd.GeoDataFrame(
            {
                "geometry": polygons,
                "class": ["tumor", "normal"],
                "classification": [
                    '{"name": "tumor", "color": [255, 0, 0]}',
                    '{"name": "normal", "color": [0, 255, 0]}',
                ],
            }
        )
        return gdf

    @pytest.fixture
    def sample_geojson_file(self, tmp_path, sample_geodataframe):
        """Create a temporary GeoJSON file for testing."""
        file_path = tmp_path / "test_annotations.geojson"
        sample_geodataframe.to_file(file_path)
        return file_path

    def test_load_from_geodataframe(self, wsi_with_annotations, sample_geodataframe):
        """Test loading annotations from a GeoDataFrame."""
        zs.io.load_annotations(
            wsi_with_annotations, sample_geodataframe, key_added="test_gdf_annotations"
        )

        assert "test_gdf_annotations" in wsi_with_annotations.shapes
        assert len(wsi_with_annotations["test_gdf_annotations"]) > 0

    def test_load_from_file_path_str(self, wsi_with_annotations, sample_geojson_file):
        """Test loading annotations from a file path as string."""
        zs.io.load_annotations(
            wsi_with_annotations,
            str(sample_geojson_file),
            key_added="test_file_str_annotations",
        )

        assert "test_file_str_annotations" in wsi_with_annotations.shapes
        assert len(wsi_with_annotations["test_file_str_annotations"]) > 0

    def test_load_from_file_path_obj(self, wsi_with_annotations, sample_geojson_file):
        """Test loading annotations from a Path object."""
        zs.io.load_annotations(
            wsi_with_annotations,
            sample_geojson_file,
            key_added="test_file_path_annotations",
        )

        assert "test_file_path_annotations" in wsi_with_annotations.shapes
        assert len(wsi_with_annotations["test_file_path_annotations"]) > 0

    @pytest.mark.parametrize("explode", [True, False])
    def test_explode_parameter(
        self, wsi_with_annotations, sample_geodataframe, explode
    ):
        """Test different explode parameter values."""
        key = f"test_explode_{explode}"
        zs.io.load_annotations(
            wsi_with_annotations, sample_geodataframe, explode=explode, key_added=key
        )

        assert key in wsi_with_annotations.shapes

    @pytest.mark.parametrize("in_bounds", [True, False])
    def test_in_bounds_parameter(
        self, wsi_with_annotations, sample_geodataframe, in_bounds
    ):
        """Test different in_bounds parameter values."""
        key = f"test_in_bounds_{in_bounds}"
        zs.io.load_annotations(
            wsi_with_annotations,
            sample_geodataframe,
            in_bounds=in_bounds,
            key_added=key,
        )

        assert key in wsi_with_annotations.shapes

    @pytest.mark.parametrize("join_with", ["tissues", ["tissues"]])
    def test_join_with_parameter(
        self, wsi_with_annotations, sample_geodataframe, join_with
    ):
        """Test different join_with parameter values."""
        key = f"test_join_with_{type(join_with).__name__}"
        zs.io.load_annotations(
            wsi_with_annotations,
            sample_geodataframe,
            join_with=join_with,
            key_added=key,
        )

        assert key in wsi_with_annotations.shapes

    @pytest.mark.parametrize(
        "json_flatten", ["classification", ["classification"], None]
    )
    def test_json_flatten_parameter(
        self, wsi_with_annotations, sample_geodataframe, json_flatten
    ):
        """Test different json_flatten parameter values."""
        key = f"test_json_flatten_{type(json_flatten).__name__}"
        zs.io.load_annotations(
            wsi_with_annotations,
            sample_geodataframe,
            json_flatten=json_flatten,
            key_added=key,
        )

        assert key in wsi_with_annotations.shapes

    @pytest.mark.parametrize("min_area", [1e2, 1e3, 1e4])
    def test_min_area_parameter(
        self, wsi_with_annotations, sample_geodataframe, min_area
    ):
        """Test different min_area parameter values."""
        key = f"test_min_area_{int(min_area)}"
        zs.io.load_annotations(
            wsi_with_annotations, sample_geodataframe, min_area=min_area, key_added=key
        )

        assert key in wsi_with_annotations.shapes

    def test_invalid_annotations_type(self, wsi_with_annotations):
        """Test error handling for invalid annotations type."""
        with pytest.raises(ValueError, match="Invalid annotations"):
            zs.io.load_annotations(wsi_with_annotations, 123)

    def test_nonexistent_file(self, wsi_with_annotations):
        """Test error handling for nonexistent file."""
        with pytest.raises(Exception):  # geopandas will raise an error
            zs.io.load_annotations(wsi_with_annotations, "nonexistent_file.geojson")

    def test_json_flatten_with_dict_data(self, wsi_with_annotations):
        """Test json_flatten with actual dictionary data."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])],
                "classification": [{"name": "tumor", "confidence": 0.95}],
            }
        )

        zs.io.load_annotations(
            wsi_with_annotations,
            gdf,
            json_flatten="classification",
            key_added="test_dict_flatten",
        )

        result = wsi_with_annotations["test_dict_flatten"]
        assert "classification_name" in result.columns
        assert "classification_confidence" in result.columns


class TestExportAnnotations:
    """Tests for the io.export_annotations function."""

    def test_export_basic(self, wsi_with_annotations):
        """Test basic export functionality."""
        result = zs.io.export_annotations(wsi_with_annotations, "annotations")

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    @pytest.mark.parametrize("in_bounds", [True, False])
    def test_in_bounds_parameter(self, wsi_with_annotations, in_bounds):
        """Test different in_bounds parameter values."""
        result = zs.io.export_annotations(
            wsi_with_annotations, "annotations", in_bounds=in_bounds
        )

        assert isinstance(result, gpd.GeoDataFrame)

    def test_export_with_classes(self, wsi_with_annotations):
        """Test export with classification column."""
        # Add a class column to the annotations
        annotations = wsi_with_annotations["annotations"].copy()
        annotations["class"] = "test_class"
        wsi_with_annotations.shapes["annotations"] = annotations

        result = zs.io.export_annotations(
            wsi_with_annotations, "annotations", classes="class"
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert "classification" in result.columns

    def test_export_with_colors_string(self, wsi_with_annotations):
        """Test export with colors as column name."""
        # Add class and color columns
        annotations = wsi_with_annotations["annotations"].copy()
        annotations["class"] = "test_class"
        annotations["color"] = "#FF0000"
        wsi_with_annotations.shapes["annotations"] = annotations

        result = zs.io.export_annotations(
            wsi_with_annotations, "annotations", classes="class", colors="color"
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert "classification" in result.columns

    def test_export_with_colors_mapping(self, wsi_with_annotations):
        """Test export with colors as mapping."""
        # Add a class column
        annotations = wsi_with_annotations["annotations"].copy()
        annotations["class"] = "test_class"
        wsi_with_annotations.shapes["annotations"] = annotations

        color_mapping = {"test_class": "#FF0000"}
        result = zs.io.export_annotations(
            wsi_with_annotations, "annotations", classes="class", colors=color_mapping
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert "classification" in result.columns

    def test_export_with_colors_sequence(self, wsi_with_annotations):
        """Test export with colors as sequence."""
        # Add a class column
        annotations = wsi_with_annotations["annotations"].copy()
        annotations["class"] = "test_class"
        wsi_with_annotations.shapes["annotations"] = annotations

        color_sequence = ["#FF0000", "#00FF00", "#0000FF"]
        result = zs.io.export_annotations(
            wsi_with_annotations, "annotations", classes="class", colors=color_sequence
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert "classification" in result.columns

    def test_export_to_file(self, wsi_with_annotations, tmp_path):
        """Test export with file saving."""
        output_file = tmp_path / "test_export.geojson"

        result = zs.io.export_annotations(
            wsi_with_annotations, "annotations", file=output_file
        )

        assert output_file.exists()
        assert isinstance(result, gpd.GeoDataFrame)

    def test_export_nonexistent_key(self, wsi_with_annotations):
        """Test error handling for nonexistent key."""
        with pytest.raises(ValueError, match="Key 'nonexistent' does not exist"):
            zs.io.export_annotations(wsi_with_annotations, "nonexistent")

    def test_export_nonexistent_classes_column(self, wsi_with_annotations):
        """Test error handling for nonexistent classes column."""
        with pytest.raises(ValueError, match="Column 'nonexistent' does not exist"):
            zs.io.export_annotations(
                wsi_with_annotations, "annotations", classes="nonexistent"
            )

    def test_export_nonexistent_colors_column(self, wsi_with_annotations):
        """Test error handling for nonexistent colors column."""
        annotations = wsi_with_annotations["annotations"].copy()
        annotations["class"] = "test_class"
        wsi_with_annotations.shapes["annotations"] = annotations

        with pytest.raises(
            ValueError, match="Color column 'nonexistent' does not exist"
        ):
            zs.io.export_annotations(
                wsi_with_annotations,
                "annotations",
                classes="class",
                colors="nonexistent",
            )

    def test_export_invalid_colors_type(self, wsi_with_annotations):
        """Test error handling for invalid colors type."""
        annotations = wsi_with_annotations["annotations"].copy()
        annotations["class"] = "test_class"
        wsi_with_annotations.shapes["annotations"] = annotations

        with pytest.raises(ValueError, match="Invalid colors"):
            zs.io.export_annotations(
                wsi_with_annotations,
                "annotations",
                classes="class",
                colors=123,  # Invalid type
            )

    def test_export_nonexistent_directory(self, wsi_with_annotations):
        """Test error handling when parent directory doesn't exist."""
        with pytest.raises(NotADirectoryError):
            zs.io.export_annotations(
                wsi_with_annotations,
                "annotations",
                file="/nonexistent/directory/file.geojson",
            )

    def test_export_format_parameter(self, wsi_with_annotations):
        """Test format parameter (currently only qupath is supported)."""
        result = zs.io.export_annotations(
            wsi_with_annotations, "annotations", format="qupath"
        )

        assert isinstance(result, gpd.GeoDataFrame)

    def test_classification_json_structure(self, wsi_with_annotations):
        """Test that classification JSON has correct structure."""
        # Add a class column
        annotations = wsi_with_annotations["annotations"].copy()
        annotations["class"] = "test_class"
        wsi_with_annotations.shapes["annotations"] = annotations

        result = zs.io.export_annotations(
            wsi_with_annotations, "annotations", classes="class"
        )

        # Check that classification column contains valid JSON
        classification_data = json.loads(result["classification"].iloc[0])
        assert "name" in classification_data
        assert "color" in classification_data
        assert classification_data["name"] == "test_class"
        assert isinstance(classification_data["color"], list)
        assert len(classification_data["color"]) == 3  # RGB values
