import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import lazyslide as zs

mpl.use("Agg")


class TestPlTissue:
    """Tests for zs.pl.tissue function."""

    def test_basic_functionality(self, wsi):
        """Test basic functionality of tissue plotting."""
        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.tissue(wsi, ax=ax)

        # Check that the plot was created
        assert isinstance(ax, Axes)
        assert len(ax.get_images()) > 0  # Should have at least one image

        plt.close(fig)

    @pytest.mark.parametrize("tissue_id", [None, 0, "all", [0, 1]])
    def test_tissue_id(self, wsi, tissue_id):
        """Test different tissue_id values."""
        # Ensure tissues are segmented
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)

        # Call the function
        if tissue_id == "all":
            # For "all", we expect a figure to be created with multiple subplots
            result = zs.pl.tissue(wsi, tissue_id=tissue_id, return_figure=True)
            # Should return a figure
            assert isinstance(result, Figure)
            axes = result.get_axes()
            assert len(axes) > 1  # Should have multiple axes
            for ax in axes:
                assert len(ax.get_images()) > 0
            plt.close(result)
        elif isinstance(tissue_id, list):
            # For a list of tissue_ids, we expect multiple axes
            result = zs.pl.tissue(wsi, tissue_id=tissue_id, return_figure=True)
            # Should return a figure
            assert isinstance(result, Figure)
            axes = result.get_axes()
            assert len(axes) == len(tissue_id)
            for ax in axes:
                assert len(ax.get_images()) > 0
        else:
            # For None or specific tissue_id, we can use a single axis
            fig = plt.figure()
            ax = fig.add_subplot(111)
            result = zs.pl.tissue(wsi, tissue_id=tissue_id, ax=ax)

            # Should return None
            assert result is None
            # Should have at least one image
            assert len(ax.get_images()) > 0

            plt.close(fig)

    @pytest.mark.parametrize("show_contours", [True, False])
    def test_show_contours(self, wsi, show_contours):
        """Test show_contours parameter."""
        # Ensure tissues are segmented
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)

        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.tissue(wsi, show_contours=show_contours, ax=ax)

        plt.close(fig)

    @pytest.mark.parametrize("show_id", [True, False])
    def test_show_id(self, wsi, show_id):
        """Test show_id parameter."""
        # Ensure tissues are segmented
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)

        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.tissue(wsi, show_id=show_id, ax=ax)

        plt.close(fig)

    @pytest.mark.parametrize("mark_origin", [True, False])
    def test_mark_origin(self, wsi, mark_origin):
        """Test mark_origin parameter."""
        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.tissue(wsi, mark_origin=mark_origin, ax=ax)

        plt.close(fig)

    @pytest.mark.parametrize("scalebar", [True, False])
    def test_scalebar(self, wsi, scalebar):
        """Test scalebar parameter."""
        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.tissue(wsi, scalebar=scalebar, ax=ax)

        plt.close(fig)

    def test_return_figure(self, wsi):
        """Test return_figure parameter."""
        # Ensure tissues are segmented
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)

        # Call with tissue_id="all" and return_figure=True
        result = zs.pl.tissue(wsi, tissue_id="all", return_figure=True)

        # Should return a figure
        assert isinstance(result, Figure)

        plt.close(result)


class TestPlTiles:
    """Tests for zs.pl.tiles function."""

    def test_basic_functionality(self, wsi):
        """Test basic functionality of tiles plotting."""
        # Ensure tissues and tiles are created
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
        if "tiles" not in wsi.shapes:
            zs.pp.tile_tissues(wsi, 256)

        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.tiles(wsi, ax=ax)

        # Check that the plot was created
        assert isinstance(ax, Axes)

        plt.close(fig)

    def test_title_string(self, wsi):
        """Regression: a str title was split into characters, showing its first."""
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
        if "tiles" not in wsi.shapes:
            zs.pp.tile_tissues(wsi, 256)
        fig, ax = plt.subplots()
        zs.pl.tiles(wsi, title="My title", ax=ax)
        assert ax.get_title() == "My title"
        plt.close(fig)

    @pytest.mark.parametrize("style", ["scatter", "heatmap"])
    def test_style(self, wsi, style):
        """Test different style values."""
        # Ensure tissues and tiles are created
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
        if "tiles" not in wsi.shapes:
            zs.pp.tile_tissues(wsi, 256)

        # Generate some features for visualization
        zs.tl.tile_prediction(wsi, model="contrast")

        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.tiles(wsi, color="contrast", style=style, ax=ax)

        plt.close(fig)

    @pytest.mark.parametrize("show_image", [True, False])
    def test_show_image(self, wsi, show_image):
        """Test show_image parameter."""
        # Ensure tissues and tiles are created
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
        if "tiles" not in wsi.shapes:
            zs.pp.tile_tissues(wsi, 256)

        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.tiles(wsi, show_image=show_image, ax=ax)

        # Check for images
        if show_image:
            assert len(ax.get_images()) > 0

        plt.close(fig)

    def test_color_feature(self, wsi):
        """Test coloring tiles by feature."""
        # Ensure tissues and tiles are created
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
        if "tiles" not in wsi.shapes:
            zs.pp.tile_tissues(wsi, 256)

        # Generate some features for visualization
        zs.tl.tile_prediction(wsi, model="contrast")
        zs.tl.tile_prediction(wsi, model="brightness")

        # Call the function with different color features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        zs.pl.tiles(wsi, color="contrast", ax=ax1)
        zs.pl.tiles(wsi, color="brightness", ax=ax2)

        plt.close(fig)

    @pytest.mark.parametrize("style", ["scatter", "heatmap"])
    @pytest.mark.parametrize(
        "palette", [{"stroma": "#0000ff"}, {"tumor": "#ff0000", "stroma": "#0000ff"}]
    )
    def test_unused_category(self, wsi, style, palette, monkeypatch):
        """Regression: a category no tile has raised KeyError if the palette
        lacked it, and otherwise painted the tiles in its color."""
        import numpy as np
        import pandas as pd
        from matplotlib.colors import to_hex

        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
        if "tiles" not in wsi.shapes:
            zs.pp.tile_tissues(wsi, 256)
        tiles = wsi["tiles"]
        labels = pd.Categorical(["stroma"] * len(tiles), categories=["tumor", "stroma"])
        monkeypatch.setitem(tiles, "tissue_type", labels)

        fig, ax = plt.subplots()
        zs.pl.tiles(
            wsi,
            color="tissue_type",
            style=style,
            palette=palette,
            show_image=False,
            ax=ax,
        )
        if style == "scatter":
            (dots,) = [c for c in ax.collections if c.get_array() is not None]
            rgba = dots.to_rgba(dots.get_array())
        else:
            px = np.concatenate(
                [np.asarray(im.get_array()).reshape(-1, 4) for im in ax.get_images()]
            )
            rgba = px[px[:, 3] > 0] / 255  # opaque cells are tiles
        plt.close(fig)
        assert {to_hex(c, keep_alpha=False) for c in rgba} == {"#0000ff"}


class TestPlAnnotations:
    """Tests for zs.pl.annotations function."""

    def test_basic_functionality(self, wsi_with_annotations):
        """Test basic functionality of annotations plotting."""
        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.annotations(wsi_with_annotations, key="annotations", ax=ax)

        # Check that the plot was created
        assert isinstance(ax, Axes)

        plt.close(fig)

    @pytest.mark.parametrize("fill", [True, False])
    def test_fill(self, wsi_with_annotations, fill):
        """Test fill parameter."""
        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.annotations(wsi_with_annotations, key="annotations", fill=fill, ax=ax)

        plt.close(fig)

    @pytest.mark.parametrize("show_image", [True, False])
    def test_show_image(self, wsi_with_annotations, show_image):
        """Test show_image parameter."""
        # Call the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        zs.pl.annotations(
            wsi_with_annotations, key="annotations", show_image=show_image, ax=ax
        )

        # Check for images
        if show_image:
            assert len(ax.get_images()) > 0

        plt.close(fig)


class TestWSIViewer:
    """Tests for zs.pl.WSIViewer class."""

    def test_basic_functionality(self, wsi):
        """Test basic functionality of WSIViewer."""
        # Create a viewer
        viewer = zs.pl.WSIViewer(wsi)

        # Add an image
        viewer.add_image()

        # Show the viewer
        fig = plt.figure()
        ax = fig.add_subplot(111)
        viewer.show(ax=ax)

        # Check that the plot was created
        assert isinstance(ax, Axes)
        assert len(ax.get_images()) > 0

        plt.close(fig)

    def test_add_scalebar(self, wsi):
        """Test adding a scalebar."""
        # Create a viewer
        viewer = zs.pl.WSIViewer(wsi)

        # Add an image and scalebar
        viewer.add_image()
        viewer.add_scalebar()

        # Show the viewer
        fig = plt.figure()
        ax = fig.add_subplot(111)
        viewer.show(ax=ax)

        plt.close(fig)

    def test_mark_origin(self, wsi):
        """Test marking the origin."""
        # Create a viewer
        viewer = zs.pl.WSIViewer(wsi)

        # Add an image and mark origin
        viewer.add_image()
        viewer.mark_origin()

        # Show the viewer
        fig = plt.figure()
        ax = fig.add_subplot(111)
        viewer.show(ax=ax)

        plt.close(fig)

    def test_add_contours(self, wsi):
        """Test adding contours."""
        # Ensure tissues are segmented
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)

        # Create a viewer
        viewer = zs.pl.WSIViewer(wsi)

        # Add an image and contours
        viewer.add_image()
        viewer.add_contours(key="tissues")

        # Show the viewer
        fig = plt.figure()
        ax = fig.add_subplot(111)
        viewer.show(ax=ax)

        # We can't reliably check for collections as they might be handled differently
        # Just verify the function runs without errors

        plt.close(fig)

    def test_add_tiles(self, wsi):
        """Test adding tiles."""
        # Ensure tissues and tiles are created
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
        if "tiles" not in wsi.shapes:
            zs.pp.tile_tissues(wsi, 256)

        # Generate some features for visualization
        zs.tl.tile_prediction(wsi, model="contrast")

        # Create a viewer
        viewer = zs.pl.WSIViewer(wsi)

        # Add an image and tiles
        viewer.add_image()
        viewer.add_tiles(key="tiles", color_by="contrast")

        # Show the viewer
        fig = plt.figure()
        ax = fig.add_subplot(111)
        viewer.show(ax=ax)

        plt.close(fig)

    def test_add_zoom(self, wsi):
        """Test adding zoom."""
        # Create a viewer
        viewer = zs.pl.WSIViewer(wsi)

        # Add an image and zoom
        viewer.add_image()
        viewer.add_zoom(0.25, 0.75, 0.25, 0.75)

        # Show the viewer
        fig = plt.figure()
        ax = fig.add_subplot(111)
        viewer.show(ax=ax)

        plt.close(fig)


class TestHeatmapGrid:
    """The heatmap mesh must land on the tiles it came from (#277)."""

    @staticmethod
    def _datasource(tile=256, stride=256, n=(4, 3), anchors=((3011, 5077),)):
        import numpy as np
        from wsidata import TileSpec

        from lazyslide.plotting._wsi_viewer import TileDataSource, Viewport

        spec = TileSpec(
            height=tile, width=tile, stride_height=stride, stride_width=stride
        )
        tiles = np.array(
            [
                (ax + i * stride, ay + j * stride)
                for ax, ay in anchors
                for i in range(n[0])
                for j in range(n[1])
            ]
        )
        ds = TileDataSource(tiles, spec)
        # A viewport whose origin and size are unrelated to the tile lattice:
        # this is what used to shift the mesh by ~1 tile.
        ds.set_viewport(Viewport(0, 0, 32914, 27615, level=0, downsample=1))
        return ds, tiles

    @staticmethod
    def _cell_corners(gy, gx, gh, gw, extent):
        """Where imshow puts the top-left corner of each cell."""
        x0, x1, y1, y0 = extent
        return x0 + gx * (x1 - x0) / gw, y0 + gy * (y1 - y0) / gh

    def test_cells_align_with_tiles(self):
        import numpy as np

        ds, tiles = self._datasource()
        layouts = list(ds.grid_layouts())
        assert len(layouts) == 1

        sel, gy, gx, gh, gw, extent = layouts[0]
        cx, cy = self._cell_corners(gy, gx, gh, gw, extent)
        assert np.allclose(cx, tiles[sel][:, 0])
        assert np.allclose(cy, tiles[sel][:, 1])
        # One cell per tile, sized as the tile.
        assert (gh, gw) == (3, 4)
        assert ((extent[1] - extent[0]) / gw, (extent[2] - extent[3]) / gh) == (
            256,
            256,
        )

    def test_separate_tissue_lattices_stay_exact(self):
        """Tiles of a second tissue sit off the first one's lattice."""
        import numpy as np

        # Anchors deliberately not a stride apart, as tissue bounding boxes are.
        ds, tiles = self._datasource(anchors=((3011, 5077), (9160, 12289)))
        layouts = list(ds.grid_layouts())
        assert len(layouts) == 2

        seen = np.zeros(len(tiles), dtype=bool)
        for sel, gy, gx, gh, gw, extent in layouts:
            cx, cy = self._cell_corners(gy, gx, gh, gw, extent)
            assert np.allclose(cx, tiles[sel][:, 0])
            assert np.allclose(cy, tiles[sel][:, 1])
            seen |= sel
        assert seen.all()

    def test_overlapping_tiles_do_not_collide(self):
        ds, tiles = self._datasource(tile=256, stride=128)
        ((_, gy, gx, _, gw, extent),) = ds.grid_layouts()

        # Using the tile size as the pitch silently mapped two tiles per cell.
        assert len({*zip(gy.tolist(), gx.tolist())}) == len(tiles)
        assert ((extent[1] - extent[0]) / gw) == 128

    def test_falls_back_to_one_grid_without_a_lattice(self):
        import numpy as np

        # Tiles that share no lattice at all, e.g. hand-made tile shapes.
        ds, tiles = self._datasource(n=(7, 6))
        assert len(tiles) > ds.MAX_LATTICES
        rng = np.random.default_rng(0)
        ds._render_tiles = ds._render_tiles + rng.integers(
            1, 256, ds._render_tiles.shape
        )
        assert len(list(ds.grid_layouts())) == 1


class TestDatashaderBackend:
    """Choosing and drawing the datashader base view of WSIViewer.add_polygons."""

    def test_plan_honours_alpha(self):
        """Regression: the plan accepted alpha but always drew opaque, and
        overlapping polygons stayed opaque once alpha was honoured."""
        pytest.importorskip("datashader")
        import geopandas as gpd
        import shapely

        from lazyslide.plotting._wsi_viewer import DatashaderFilledPolygonRenderPlan

        class Image:
            def get_extent(self):
                return (0, 100, 100, 0)

            def get_image_size(self):
                return (100, 100)

        gdf = gpd.GeoDataFrame(
            {
                "geometry": [shapely.box(10, 10, 50, 50), shapely.box(30, 30, 70, 70)],
                "c": ["a", "a"],
            }
        )
        fig, ax = plt.subplots()
        DatashaderFilledPolygonRenderPlan(
            gdf, Image(), color_by="c", palette={"a": "#ff0000"}, alpha=0.5
        ).render(ax)
        alpha = ax.images[0].get_array()[..., 3]
        assert set(alpha[alpha > 0].tolist()) == {128}
        plt.close(fig)

    def test_missing_datashader_falls_back(self, wsi_no_spec, monkeypatch):
        """Regression: the fallback still imported datashader and crashed."""
        import sys

        from lazyslide.plotting._wsi_viewer import DatashaderFilledPolygonRenderPlan

        monkeypatch.setitem(sys.modules, "datashader", None)  # as if not installed
        viewer = zs.pl.WSIViewer(wsi_no_spec)
        with pytest.warns(UserWarning, match="not installed"):
            viewer.add_polygons("no_spec_tiles", backend="datashader")
        assert not any(
            isinstance(p, DatashaderFilledPolygonRenderPlan)
            for p in viewer._render_plans
        )

    def test_matplotlib_backend_wins_over_polygon_count(self, wsi_no_spec):
        """Regression: backend="matplotlib" was ignored above 10,000 polygons."""
        import geopandas as gpd
        import numpy as np
        import shapely
        from wsidata.io import add_shapes

        from lazyslide.plotting._wsi_viewer import DatashaderFilledPolygonRenderPlan

        grid = np.arange(101) * 15 + 10
        x, y = (a.ravel() for a in np.meshgrid(grid, grid))
        boxes = gpd.GeoDataFrame({"geometry": shapely.box(x, y, x + 10, y + 10)})
        add_shapes(wsi_no_spec, "many_polygons", boxes)
        viewer = zs.pl.WSIViewer(wsi_no_spec)
        viewer.add_polygons("many_polygons", backend="matplotlib")
        assert not any(
            isinstance(p, DatashaderFilledPolygonRenderPlan)
            for p in viewer._render_plans
        )
