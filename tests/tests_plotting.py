import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import lazyslide as zs


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

    @pytest.mark.parametrize("style", ["scatter", "heatmap"])
    def test_style(self, wsi, style):
        """Test different style values."""
        # Ensure tissues and tiles are created
        if "tissues" not in wsi.shapes:
            zs.pp.find_tissues(wsi)
        if "tiles" not in wsi.shapes:
            zs.pp.tile_tissues(wsi, 256)

        # Generate some features for visualization
        zs.pp.score_tiles(wsi, scorers=["contrast"])

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
        zs.pp.score_tiles(wsi, scorers=["contrast", "brightness"])

        # Call the function with different color features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        zs.pl.tiles(wsi, color="contrast", ax=ax1)
        zs.pl.tiles(wsi, color="brightness", ax=ax2)

        plt.close(fig)


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
        zs.pp.score_tiles(wsi, scorers=["contrast"])

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
