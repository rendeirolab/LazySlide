import numpy as np
import pytest
from shapely.geometry import Polygon

import lazyslide as zs


class TestPPFindTissues:
    """Tests for the pp.find_tissues function."""

    @pytest.mark.parametrize("detect_holes", [True, False])
    @pytest.mark.parametrize("key_added", ["tissue", "tissue2"])
    def test_basic_functionality(self, wsi, detect_holes, key_added):
        """Test basic functionality with different detect_holes and key_added values."""
        zs.pp.find_tissues(wsi, detect_holes=detect_holes, key_added=key_added)

        # Check if the key is added to wsi.shapes
        assert key_added in wsi.shapes

        # Check if there are no interiors when detect_holes=False
        if not detect_holes:
            tissue = wsi[key_added].geometry[0]
            assert len(tissue.interiors) == 0

    @pytest.mark.parametrize("level", ["auto", 2, -1])
    def test_level_parameter(self, wsi, level):
        """Test different level parameter values."""
        key = f"tissue_level_{level}"
        zs.pp.find_tissues(wsi, level=level, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

        # Check if tissues were found
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("refine_level", [None, "auto", 0])
    def test_refine_level_parameter(self, wsi, refine_level):
        """Test different refine_level parameter values."""
        key = f"tissue_refine_{refine_level}"
        zs.pp.find_tissues(wsi, refine_level=refine_level, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

        # Check if tissues were found
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("to_hsv", [True, False])
    def test_to_hsv_parameter(self, wsi, to_hsv):
        """Test different to_hsv parameter values."""
        key = f"tissue_hsv_{to_hsv}"
        zs.pp.find_tissues(wsi, to_hsv=to_hsv, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

        # Check if tissues were found
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("blur_ksize", [5, 17, 31])
    def test_blur_ksize_parameter(self, wsi, blur_ksize):
        """Test different blur_ksize parameter values."""
        key = f"tissue_blur_{blur_ksize}"
        zs.pp.find_tissues(wsi, blur_ksize=blur_ksize, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

        # Check if tissues were found
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("threshold", [3, 7, 15])
    def test_threshold_parameter(self, wsi, threshold):
        """Test different threshold parameter values."""
        key = f"tissue_threshold_{threshold}"
        zs.pp.find_tissues(wsi, threshold=threshold, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

        # Check if tissues were found
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("morph_n_iter", [1, 3, 5])
    def test_morph_n_iter_parameter(self, wsi, morph_n_iter):
        """Test different morph_n_iter parameter values."""
        key = f"tissue_morph_iter_{morph_n_iter}"
        zs.pp.find_tissues(wsi, morph_n_iter=morph_n_iter, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

        # Check if tissues were found
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("morph_ksize", [3, 7, 11])
    def test_morph_ksize_parameter(self, wsi, morph_ksize):
        """Test different morph_ksize parameter values."""
        key = f"tissue_morph_ksize_{morph_ksize}"
        zs.pp.find_tissues(wsi, morph_ksize=morph_ksize, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

        # Check if tissues were found
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("min_tissue_area", [1e-4, 1e-3, 1e-2])
    def test_min_tissue_area_parameter(self, wsi, min_tissue_area):
        """Test different min_tissue_area parameter values."""
        key = f"tissue_min_area_{min_tissue_area}"
        zs.pp.find_tissues(wsi, min_tissue_area=min_tissue_area, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

    @pytest.mark.parametrize("min_hole_area", [1e-6, 1e-5, 1e-4])
    def test_min_hole_area_parameter(self, wsi, min_hole_area):
        """Test different min_hole_area parameter values."""
        key = f"tissue_min_hole_{min_hole_area}"
        zs.pp.find_tissues(wsi, min_hole_area=min_hole_area, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

    @pytest.mark.parametrize("filter_artifacts", [True, False])
    def test_filter_artifacts_parameter(self, wsi, filter_artifacts):
        """Test different filter_artifacts parameter values."""
        key = f"tissue_filter_{filter_artifacts}"
        zs.pp.find_tissues(wsi, filter_artifacts=filter_artifacts, key_added=key)

        # Check if the key is added to wsi.shapes
        assert key in wsi.shapes

        # Check if tissues were found
        assert len(wsi[key]) > 0

    def test_tissue_geometry_properties(self, wsi):
        """Test properties of the tissue geometries."""
        key = "tissue_geometry_test"
        zs.pp.find_tissues(wsi, key_added=key)

        # Check if tissues were found
        assert len(wsi[key]) > 0

        # Check if all geometries are Polygons
        for geom in wsi[key].geometry:
            assert isinstance(geom, Polygon)

        # Check if all geometries have area > 0
        for geom in wsi[key].geometry:
            assert geom.area > 0

        # Check if all geometries are valid
        for geom in wsi[key].geometry:
            assert geom.is_valid


class TestPPTileTissues:
    """Tests for the pp.tile_tissues function."""

    def test_tile_px(self, wsi):
        """Test basic functionality with default parameters."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles")

        # Check if tiles were created
        assert "tiles" in wsi.shapes
        assert len(wsi["tiles"]) > 0

    def test_mpp(self, wsi):
        """Test with mpp parameter."""
        zs.pp.tile_tissues(wsi, 256, mpp=1, key_added="tiles1")

        # Check if tiles were created
        assert "tiles1" in wsi.shapes
        assert len(wsi["tiles1"]) > 0

    @pytest.mark.xfail(raises=ValueError)
    def test_slide_mpp(self, wsi):
        """Test that slide_mpp parameter raises ValueError."""
        zs.pp.tile_tissues(wsi, 256, slide_mpp=1, key_added="tiles2")

    @pytest.mark.parametrize("tile_size", [128, 256, (256, 128)])
    def test_tile_size(self, wsi, tile_size):
        """Test different tile sizes."""
        # Create a safe key name
        if isinstance(tile_size, tuple):
            key = f"tiles_size_{tile_size[0]}x{tile_size[1]}"
        else:
            key = f"tiles_size_{tile_size}"
        zs.pp.tile_tissues(wsi, tile_size, key_added=key)

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("stride_px", [None, 128, (128, 64)])
    def test_stride_px(self, wsi, stride_px):
        """Test different stride_px values."""
        # Create a safe key name
        if isinstance(stride_px, tuple):
            key = f"tiles_stride_{stride_px[0]}x{stride_px[1]}"
        elif stride_px is None:
            key = "tiles_stride_None"
        else:
            key = f"tiles_stride_{stride_px}"
        zs.pp.tile_tissues(wsi, 256, stride_px=stride_px, key_added=key)

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("overlap", [None, 0.25, 64])
    def test_overlap(self, wsi, overlap):
        """Test different overlap values."""
        key = f"tiles_overlap_{overlap}"
        zs.pp.tile_tissues(wsi, 256, overlap=overlap, key_added=key)

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("edge", [True, False])
    def test_edge(self, wsi, edge):
        """Test different edge values."""
        key = f"tiles_edge_{edge}"
        zs.pp.tile_tissues(wsi, 256, edge=edge, key_added=key)

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("ops_level", [None, 0, 1])
    def test_ops_level(self, wsi, ops_level):
        """Test different ops_level values."""
        # Create a safe key name
        if ops_level is None:
            key = "tiles_ops_level_None"
        else:
            key = f"tiles_ops_level_{ops_level}"

        # When ops_level is specified, mpp must also be specified
        # Use a higher mpp value for higher ops_level to avoid errors
        if ops_level == 0:
            mpp = 0.5
        elif ops_level == 1:
            mpp = 2.0  # Higher mpp for level 1
        else:
            mpp = None

        zs.pp.tile_tissues(wsi, 256, ops_level=ops_level, mpp=mpp, key_added=key)

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("background_filter", [True, False])
    def test_background_filter(self, wsi, background_filter):
        """Test different background_filter values."""
        key = f"tiles_bg_filter_{background_filter}"
        zs.pp.tile_tissues(wsi, 256, background_filter=background_filter, key_added=key)

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("background_fraction", [0.1, 0.3, 0.5])
    def test_background_fraction(self, wsi, background_fraction):
        """Test different background_fraction values."""
        key = f"tiles_bg_frac_{background_fraction}"
        zs.pp.tile_tissues(
            wsi, 256, background_fraction=background_fraction, key_added=key
        )

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    @pytest.mark.parametrize("background_filter_mode", ["approx", "exact"])
    def test_background_filter_mode(self, wsi, background_filter_mode):
        """Test different background_filter_mode values."""
        key = f"tiles_bg_mode_{background_filter_mode}"
        zs.pp.tile_tissues(
            wsi, 256, background_filter_mode=background_filter_mode, key_added=key
        )

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

        # If the mode is exact, check all polygons intersect with tissue
        if background_filter_mode == "exact":
            tissue = wsi["tissues"].geometry.union_all()
            assert all(geom.intersects(tissue) for geom in wsi[key].geometry)

    def test_no_tissue_key(self, wsi):
        """Test behavior when tissue_key is None."""
        key = "tiles_no_tissue"
        zs.pp.tile_tissues(wsi, 256, tissue_key=None, key_added=key)

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    def test_invalid_tissue_key(self, wsi):
        """Test behavior with an invalid tissue_key."""
        key = "tiles_invalid_tissue"
        zs.pp.tile_tissues(wsi, 256, tissue_key="nonexistent_tissue", key_added=key)

        # Check if tiles were created (should fall back to whole slide)
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

    def test_return_tiles(self, wsi):
        """Test return_tiles parameter."""
        result = zs.pp.tile_tissues(
            wsi, 256, key_added="tiles_return", return_tiles=True
        )

        # Check if result is a tuple with two elements
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Check if tiles were created
        assert "tiles_return" in wsi.shapes
        assert len(wsi["tiles_return"]) > 0

        # Check if returned tiles match what's in wsi.shapes
        tiles_df, tile_spec = result
        assert len(tiles_df) == len(wsi["tiles_return"])

    def test_tile_geometry_properties(self, wsi):
        """Test properties of the tile geometries."""
        key = "tiles_geometry_test"
        zs.pp.tile_tissues(wsi, 256, key_added=key)

        # Check if tiles were created
        assert key in wsi.shapes
        assert len(wsi[key]) > 0

        # Check if all geometries are Polygons
        for geom in wsi[key].geometry:
            assert isinstance(geom, Polygon)

        # Check if all geometries have area > 0
        for geom in wsi[key].geometry:
            assert geom.area > 0

        # Check if all geometries are valid
        for geom in wsi[key].geometry:
            assert geom.is_valid

        # Check if all tiles have the same area (for square tiles)
        areas = [geom.area for geom in wsi[key].geometry]
        assert np.allclose(areas, areas[0])


class TestPPTileGraph:
    """Tests for the pp.tile_graph function."""

    def test_basic_functionality(self, wsi):
        """Test basic functionality with default parameters."""
        # First create tiles to build graph from
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_for_graph")

        # Test the tile_graph function
        zs.pp.tile_graph(wsi, tile_key="tiles_for_graph")

        # Check if the graph table was created (Key.tile_graph format: {name}_graph)
        expected_key = "tiles_for_graph_graph"
        assert expected_key in wsi.tables

        # Check if the table has the expected structure
        table = wsi[expected_key]
        assert hasattr(table, "obsp")
        assert "spatial_connectivities" in table.obsp
        assert "spatial_distances" in table.obsp
        assert "spatial" in table.uns

    @pytest.mark.parametrize("n_neighs", [4, 6, 8])
    def test_n_neighs_parameter(self, wsi, n_neighs):
        """Test different n_neighs parameter values."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_neighs")

        table_key = f"graph_neighs_{n_neighs}"
        zs.pp.tile_graph(
            wsi, n_neighs=n_neighs, tile_key="tiles_neighs", table_key=table_key
        )

        # Check if the graph table was created
        assert table_key in wsi.tables

        # Check if the table has the expected structure
        table = wsi[table_key]
        assert table.uns["spatial"]["params"]["n_neighbors"] == n_neighs

    @pytest.mark.parametrize("n_rings", [1, 2, 3])
    def test_n_rings_parameter(self, wsi, n_rings):
        """Test different n_rings parameter values."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_rings")

        table_key = f"graph_rings_{n_rings}"
        zs.pp.tile_graph(
            wsi, n_rings=n_rings, tile_key="tiles_rings", table_key=table_key
        )

        # Check if the graph table was created
        assert table_key in wsi.tables

        # Check if the table has the expected structure
        table = wsi[table_key]
        assert hasattr(table, "obsp")
        assert "spatial_connectivities" in table.obsp

    @pytest.mark.parametrize("delaunay", [True, False])
    def test_delaunay_parameter(self, wsi, delaunay):
        """Test different delaunay parameter values."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_delaunay")

        table_key = f"graph_delaunay_{delaunay}"
        zs.pp.tile_graph(
            wsi, delaunay=delaunay, tile_key="tiles_delaunay", table_key=table_key
        )

        # Check if the graph table was created
        assert table_key in wsi.tables

        # Check if the table has the expected structure
        table = wsi[table_key]
        assert hasattr(table, "obsp")
        assert "spatial_connectivities" in table.obsp

    @pytest.mark.parametrize("transform", [None, "cosine"])
    def test_transform_parameter(self, wsi, transform):
        """Test different transform parameter values."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_transform")

        table_key = f"graph_transform_{transform}"
        zs.pp.tile_graph(
            wsi, transform=transform, tile_key="tiles_transform", table_key=table_key
        )

        # Check if the graph table was created
        assert table_key in wsi.tables

        # Check if the table has the expected structure
        table = wsi[table_key]
        assert table.uns["spatial"]["params"]["transform"] == transform

    @pytest.mark.parametrize("set_diag", [True, False])
    def test_set_diag_parameter(self, wsi, set_diag):
        """Test different set_diag parameter values."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_diag")

        table_key = f"graph_diag_{set_diag}"
        zs.pp.tile_graph(
            wsi, set_diag=set_diag, tile_key="tiles_diag", table_key=table_key
        )

        # Check if the graph table was created
        assert table_key in wsi.tables

        # Check if the table has the expected structure
        table = wsi[table_key]
        connectivities = table.obsp["spatial_connectivities"]

        if set_diag:
            # Check that diagonal elements are set to 1
            assert (connectivities.diagonal() == 1).all()
        else:
            # Check that diagonal elements are 0
            assert (connectivities.diagonal() == 0).all()

    def test_custom_tile_key(self, wsi):
        """Test with custom tile_key parameter."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="custom_tiles")

        zs.pp.tile_graph(wsi, tile_key="custom_tiles")

        # Check if the graph table was created with the correct key (Key.tile_graph format: {name}_graph)
        expected_key = "custom_tiles_graph"
        assert expected_key in wsi.tables

    def test_custom_table_key(self, wsi):
        """Test with custom table_key parameter."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_custom_table")

        custom_table_key = "my_custom_graph"
        zs.pp.tile_graph(wsi, tile_key="tiles_custom_table", table_key=custom_table_key)

        # Check if the graph table was created with the custom key
        assert custom_table_key in wsi.tables

    def test_graph_properties(self, wsi):
        """Test properties of the created graph."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_properties")

        zs.pp.tile_graph(wsi, tile_key="tiles_properties")

        # Get the created table (Key.tile_graph format: {name}_graph)
        expected_key = "tiles_properties_graph"
        table = wsi[expected_key]

        # Check that connectivities and distances are sparse matrices
        connectivities = table.obsp["spatial_connectivities"]
        distances = table.obsp["spatial_distances"]

        from scipy.sparse import issparse

        assert issparse(connectivities)
        assert issparse(distances)

        # Check that dimensions match the number of tiles
        n_tiles = len(wsi["tiles_properties"])
        assert connectivities.shape == (n_tiles, n_tiles)
        assert distances.shape == (n_tiles, n_tiles)

        # Check that distances are non-negative
        assert (distances.data >= 0).all()

    def test_invalid_transform(self, wsi):
        """Test that invalid transform raises NotImplementedError."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_invalid")

        with pytest.raises(NotImplementedError):
            zs.pp.tile_graph(
                wsi, transform="invalid_transform", tile_key="tiles_invalid"
            )

    def test_missing_tile_key(self, wsi):
        """Test behavior when tile_key doesn't exist."""
        # This should raise an error when trying to access non-existent tiles
        with pytest.raises(KeyError):
            zs.pp.tile_graph(wsi, tile_key="nonexistent_tiles")

    def test_existing_table_key(self, wsi):
        """Test behavior when table_key already exists."""
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 256, key_added="tiles_existing")

        table_key = "existing_graph"

        # First call to create the table
        zs.pp.tile_graph(wsi, tile_key="tiles_existing", table_key=table_key)
        assert table_key in wsi.tables

        # Second call should update the existing table
        zs.pp.tile_graph(
            wsi, tile_key="tiles_existing", table_key=table_key, n_neighs=8
        )
        assert table_key in wsi.tables

        # Check that parameters were updated
        table = wsi[table_key]
        assert table.uns["spatial"]["params"]["n_neighbors"] == 8
