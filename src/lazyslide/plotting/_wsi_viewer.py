from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from itertools import cycle
from numbers import Number
from typing import Any, Dict, List, Literal, Sequence, Union

import cv2
import numpy as np
import pandas as pd
from geopandas import points_from_xy
from legendkit import cat_legend, colorart, vstack
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.colors import ListedColormap, is_color_like, to_rgba
from matplotlib.patches import Patch, Rectangle
from matplotlib.typing import ColorType
from shapely import MultiPolygon, Polygon, box
from wsidata import TileSpec, WSIData
from wsidata.reader import ReaderBase

from .._utils import find_stack_level

LAZYSLIDE_PALETTE = (
    "#e60049",
    "#0bb4ff",
    "#50e991",
    "#e6d800",
    "#9b19f5",
    "#ffa300",
    "#dc0ab4",
    "#b3d4ff",
    "#00bfa0",
)

PaletteType = Union[Dict, Sequence[ColorType], ColorType]


@dataclass
class Viewport:
    """
    The dataclass to store view port of the slide.

    Attributes
    ----------
    x : int
        The x-coordinate of the window.
    y : int
        The y-coordinate of the window.
    w : int
        The width of the window.
    h : int
        The height of the window.
    level : int
        The level of the pyramid to view.
    downsample : int
        The downsample factor of the window.

    """

    x: int
    y: int
    w: int
    h: int
    level: int
    downsample: int

    @cached_property
    def box(self) -> Polygon:
        """Return the window in shapely polygon."""

        return box(
            self.x,
            self.y,
            self.x + self.w * self.downsample,
            self.y + self.h * self.downsample,
        )


class DataSource:
    """The base class for data sources.

    Attributes
    ----------
    viewport : Viewport
        The current viewport.
    _sel : np.ndarray
        The selection mask of the data depending on the viewport.
    _sel_attrs : Dict
        The selection of data depending on the viewport.

    """

    viewport: Viewport | None = None
    _sel: np.ndarray[Any, np.dtype[np.bool_]] | None = None
    _sel_attrs: Dict = {}

    def set_viewport_hook(self):
        """
        The hook that runs if the viewport is updated.
        """
        pass

    def set_viewport(self, viewport: Viewport):
        if self.viewport == viewport:
            return
        self.viewport = viewport
        self.set_viewport_hook()

    def set_data(self, **kwargs):
        """
        If there are data that depends on the viewport, set them here.
        """
        for k, v in kwargs.items():
            if v is not None:
                v = np.asarray(v)
            self._sel_attrs.update({k: v})

    def get_data(self, k):
        """
        Get the data that depends on the viewport during rendering.
        """
        d = self._sel_attrs.get(k)
        if (d is not None) & (self._sel is not None):
            return d[self._sel]
        else:
            return d

    @property
    def sel(self):
        return self._sel


class ImageDataSource(DataSource):
    def __init__(self, reader: ReaderBase):
        self.reader = reader
        self._image = None
        self._refresh = False

    def set_viewport_hook(self):
        self._refresh = True

    @property
    def image(self):
        # Must be lazy evaluated at render time to avoid loading the image multiple times
        if self._image is None or self._refresh:
            x, y, w, h = (
                self.viewport.x,
                self.viewport.y,
                self.viewport.w,
                self.viewport.h,
            )
            self._image = self.reader.get_region(x, y, w, h, level=self.viewport.level)
            self._image[np.where(self._image == 0)] = 255
            self._refresh = False
        return self._image

    def get_image_size(self) -> tuple[int, int]:
        """Return the width, height of the actual image, not the data."""
        downsample = self.viewport.downsample
        return int(self.viewport.w * downsample), int(self.viewport.h * downsample)

    def get_extent(self):
        x, y = self.viewport.x, self.viewport.y
        w, h = self.get_image_size()

        return [x, x + w, y + h, y]


class TileDataSource(DataSource):
    def __init__(self, tiles: np.ndarray, tile_spec: TileSpec):
        self._tiles = tiles
        # Place the points at the center of the tiles
        self._points = points_from_xy(tiles[:, 0], tiles[:, 1])
        self.tile_spec = tile_spec
        self._sel = np.ones_like(tiles, dtype=bool)
        self._render_tiles = tiles

    def set_viewport_hook(self):
        box = self.viewport.box
        # scale_f = 1 / self.viewport.downsample

        render_tiles = self._points
        # render_tiles = render_tiles.scale(xfact=scale_f, yfact=scale_f, origin=(0, 0))
        sel = box.intersects(render_tiles)
        render_tiles = render_tiles[sel]
        self._sel = sel
        # Get the xy coordinates in numpy array
        self._render_tiles = np.ceil(
            np.asarray([[p.x, p.y] for p in render_tiles])
        ).astype(int)

    @property
    def tiles(self):
        return self._render_tiles

    @property
    def heatmap_tiles(self):
        return (
            (self._render_tiles - (self.viewport.x, self.viewport.y))
            / self.viewport.downsample
        ).astype(int)

    @property
    def tiles_center(self):
        """Return the center of the tiles at level 0."""
        tiles = self._render_tiles.copy()
        tile_width, tile_height = self.tile_shape_base
        tiles[:, 0] = tiles[:, 0] + int(tile_width // 2)
        tiles[:, 1] = tiles[:, 1] + int(tile_height // 2)
        return tiles

    @property
    def tile_shape(self) -> tuple[int, int]:
        """The W, H of the tile in the viewport, after downsample."""
        width = int(self.tile_spec.base_width // self.viewport.downsample)
        height = int(self.tile_spec.base_height // self.viewport.downsample)
        return width, height

    @property
    def tile_shape_base(self) -> tuple[int, int]:
        """The W, H of the tile at level 0."""
        return self.tile_spec.base_width, self.tile_spec.base_height


class PolygonDataSource(DataSource):
    def __init__(self, polygons: List[Polygon]):
        self._polygons = polygons
        self._render_polygons = polygons
        self._sel = np.ones_like(polygons, dtype=bool)

    def set_viewport_hook(self):
        box = self.viewport.box
        # scale_f = 1 / self.viewport.downsample
        # render_polygons = np.asarray([scale(p, xfact=scale_f, yfact=scale_f, origin=(0, 0)) for p in self._polygons])
        sel = box.intersects(self._polygons)
        self._render_polygons = self._polygons[sel]
        self._sel = sel

    @property
    def polygons(self):
        return self._render_polygons


class RenderPlan:
    """
    The base class for rendering plans.

    Attributes
    ----------
    datasource : DataSource
        The data source for the rendering plan.
    zoom_view_visible : bool
        Whether the plan is rendered in the zoom view.

    """

    datasource: DataSource
    zoom_view_visible: bool = True
    legend_visible: bool = True
    on_zoom_view: bool = False
    legend: Artist | None = None

    def render(self, ax):
        """The rendering logics of the plan."""
        pass

    def get_legend(self) -> Artist | None:
        """Return the legend of the plan."""
        return self.legend


class SlideImageRenderPlan(RenderPlan):
    """
    The rendering plan for the slide image.

    Parameters
    ----------
    datasource : ImageDataSource
        The image data source for the rendering plan.

    """

    def __init__(self, datasource: ImageDataSource):
        self.datasource: ImageDataSource = datasource

    def render(self, ax):
        image = self.datasource.image
        extent = self.datasource.get_extent()
        ax.imshow(image, extent=extent, origin="upper", zorder=-100)


class ScaleBarRenderPlan(RenderPlan):
    def __init__(self, datasource: ImageDataSource, dx, units="um", **kwargs):
        self.datasource: ImageDataSource = datasource
        self.dx = dx
        self.units = units
        self.kwargs = kwargs

    def render(self, ax):
        from matplotlib_scalebar.scalebar import ScaleBar

        scalebar = ScaleBar(self.dx, units=self.units, **self.kwargs)
        ax.add_artist(scalebar)


class OriginXYArrowRenderPlan(RenderPlan):
    def __init__(self, length=30, linewidth=1, color="k", **kwargs):
        self.length = length
        self.kwargs = kwargs
        self._arrow_props = dict(
            arrowstyle="<|-", shrinkA=0, shrinkB=0, color=color, linewidth=linewidth
        )
        self._annotate_props = dict(
            xy=(0, 1),
            xycoords="axes fraction",
            textcoords="offset points",
            va="center",
            ha="center",
            arrowprops=self._arrow_props,
        )

    def render(self, ax):
        ax.annotate("x", xytext=(self.length, 0), **self._annotate_props)
        ax.annotate("y", xytext=(0, -self.length), **self._annotate_props)


class HeatmapTilesRenderPlan(RenderPlan):
    def __init__(
        self,
        tile_datasource: TileDataSource,
        image_datasource: ImageDataSource,
        values: np.ndarray,
        palette: Dict = None,
        cmap="coolwarm",
        norm=None,
        vmin=None,
        vmax=None,
        alpha=0.7,
        smooth=False,
        smooth_scale=2,
        legend_kws=None,
        **kwargs: Any,  # noqa: ANN001
    ):
        self.datasource: TileDataSource = tile_datasource
        self.image_datasource: ImageDataSource = image_datasource
        # If is categorical
        if palette is not None:
            # encode values into numerical and create a cmap from palette
            values = pd.Categorical(values)
            cmap = ListedColormap([palette[c] for c in values.categories])
            values = values.codes

        self.datasource.set_data(values=values)
        self.palette = palette
        self.cmap = cmap
        self.norm = norm
        self.vmin = vmin
        self.vmax = vmax
        self.alpha = alpha
        self.smooth = smooth
        self.smooth_scale = smooth_scale
        self.legend_kws = legend_kws or {}
        self._A = None

    def render(self, ax):
        w, h = self.image_datasource.viewport.w, self.image_datasource.viewport.h
        tile_image = np.full((h, w), np.nan)
        tiles = self.datasource.heatmap_tiles
        tile_width, tile_height = self.datasource.tile_shape
        vs = self.datasource.get_data("values")

        for (x, y), v in zip(tiles, vs):
            tile_image[y : y + tile_height + 1, x : x + tile_width + 1] = v

        cmap = get_cmap(self.cmap)
        sm = ScalarMappable(norm=self.norm, cmap=cmap)
        sm.set_clim(self.vmin, self.vmax)
        sm.set_array(tile_image)
        # sm.autoscale()

        A = sm.to_rgba(tile_image, bytes=True, alpha=self.alpha)

        # Apply gaussian blur to the heatmap
        # kernel size is 2x of tile size
        if self.smooth:
            ksize = int(self.smooth_scale * max(self.datasource.tile_shape))
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            A = cv2.GaussianBlur(A, (ksize, ksize), 0)

        # Downsample and then upscale to the original size
        # down = cv2.resize(A, (A.shape[1] // 4, A.shape[0] // 4), interpolation=cv2.INTER_AREA)
        # A = cv2.resize(down, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_CUBIC)

        self._A = A
        ax.imshow(
            A,
            extent=self.image_datasource.get_extent(),
            origin="upper",
            zorder=-99,
            interpolation="none",
        )
        if not self.on_zoom_view:
            if self.palette is not None:
                self.legend = cat_legend(
                    colors=self.palette.values(),
                    labels=self.palette.keys(),
                    **self.legend_kws,
                )
            else:
                self.legend = colorart(sm, **self.legend_kws)


class ScatterTilesRenderPlan(RenderPlan):
    def __init__(
        self,
        datasource: TileDataSource,
        values: np.ndarray,
        palette: Dict = None,
        cmap="coolwarm",
        norm=None,
        vmin=None,
        vmax=None,
        alpha=1.0,
        size=None,
        zoom_size=None,
        marker="o",
        rasterized=True,
        legend_kws=None,
        **kwargs: Any,  # noqa: ANN001
    ):
        self.datasource: TileDataSource = datasource
        # If is categorical
        if palette is not None:
            # encode values into numerical and create a cmap from palette
            values = pd.Categorical(values)
            cmap = ListedColormap([palette[c] for c in values.categories])
            values = values.codes

        self.datasource.set_data(values=values)
        self.palette = palette
        self.cmap = cmap
        self.norm = norm
        self.vmin = vmin
        self.vmax = vmax
        self.alpha = alpha
        self.size = size
        self.zoom_size = zoom_size
        self.marker = marker
        self.rasterized = rasterized
        self.legend_kws = legend_kws or {}

    def render(self, ax):
        tiles = self.datasource.tiles_center
        values = self.datasource.get_data("values")
        # Determine the size of the scatter points
        s = self.zoom_size if self.on_zoom_view else self.size
        if s is None:
            s = 12000 / len(tiles)
        sm = ax.scatter(
            tiles[:, 0],
            tiles[:, 1],
            c=values,
            cmap=self.cmap,
            norm=self.norm,
            vmin=self.vmin,
            vmax=self.vmax,
            s=s,
            marker=self.marker,
            alpha=self.alpha,
            rasterized=self.rasterized,
        )
        if not self.on_zoom_view:
            if self.palette is not None:
                self.legend = cat_legend(
                    colors=self.palette.values(),
                    labels=self.palette.keys(),
                    **self.legend_kws,
                )
            else:
                self.legend = colorart(sm, **self.legend_kws)


class GridTilesRenderPlan(RenderPlan):
    def __init__(
        self,
        datasource: TileDataSource,
        color="k",
        linewidth=0.1,
    ):
        self.datasource: TileDataSource = datasource
        self.color = color
        self.linewidth = linewidth

    def render(self, ax):
        width, height = self.datasource.tile_shape_base
        patches = []
        for x, y in self.datasource.tiles:
            patches.append(Rectangle(xy=(x, y), width=width, height=height))
        ax.add_collection(
            PatchCollection(
                patches, facecolor="none", edgecolor=self.color, lw=self.linewidth
            )
        )


class PolygonMixin(RenderPlan):
    def __init__(
        self,
        datasource: PolygonDataSource,
        **data,
    ):
        self.datasource: PolygonDataSource = datasource
        self.datasource.set_data(**data)

    @staticmethod
    def _label_patch(ax, patch: Patch, name, pad=0.1, box_color="white", **kwargs):
        kwargs = {} if kwargs is None else kwargs
        options = dict(
            color="black",
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(facecolor=box_color, pad=2, lw=1),
        )
        options.update(kwargs)
        # Check if patch is inside the axis limits
        xrange = np.sort(ax.get_xlim())
        yrange = np.sort(ax.get_ylim())
        extent = patch.get_extents()
        extent = patch.get_transform().inverted().transform_bbox(extent)
        in_x = extent.x0 >= xrange[0] and extent.x1 <= xrange[1]
        in_y = extent.y0 >= yrange[0] and extent.y1 <= yrange[1]
        if not in_x or not in_y:
            return
        ax.annotate(name, (0.5, 1 + pad), xycoords=patch, **options)

    @staticmethod
    def _filled_polygon_patch(polygon: Polygon, **kwargs):
        """
        Create a matplotlib patch from a shapely polygon.

        """
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        path = Path.make_compound_path(
            Path(np.asarray(polygon.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
        )
        return PathPatch(path, **kwargs)

    @staticmethod
    def _contour_polygon_patch(polygon: Polygon, outline_kwargs=None, hole_kwargs=None):
        """
        Create a matplotlib patch from a shapely polygon.

        """
        from matplotlib.patches import Polygon as PolygonPatch

        outline_kwargs = {} if outline_kwargs is None else outline_kwargs
        hole_kwargs = {} if hole_kwargs is None else hole_kwargs

        outer = np.asarray(polygon.exterior.coords)[:, :2]
        inner = [np.asarray(ring.coords)[:, :2] for ring in polygon.interiors]

        return PolygonPatch(outer, **outline_kwargs), [
            PolygonPatch(h, **hole_kwargs) for h in inner
        ]

    @staticmethod
    def _bbox_polygon_patch(polygon: Polygon, **kwargs):
        """
        Create a matplotlib patch from the bbox of a shapely polygon.

        """
        from matplotlib.patches import Rectangle

        xmin, ymin, xmax, ymax = polygon.bounds
        return Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)


class ContourRenderPlan(PolygonMixin):
    def __init__(
        self,
        polygons: PolygonDataSource,
        labels: Sequence = None,
        colors: Sequence = None,
        palette: Dict = None,
        outline_color: ColorType = "#117554",
        hole_color: ColorType = "#4379F2",
        linewidth: int = 1,
        outline_kws=None,
        hole_kws=None,
        legend_kws=None,
    ):
        self.palette = palette
        super().__init__(polygons, labels=labels, colors=colors)

        outline_kws = {} if outline_kws is None else outline_kws
        hole_kws = {} if hole_kws is None else hole_kws

        self.outline_kws = dict(
            edgecolor=outline_color, linewidth=linewidth, fill=False
        )
        self.outline_kws.update(outline_kws)
        self.hole_kws = dict(edgecolor=hole_color, linewidth=linewidth, fill=False)
        self.hole_kws.update(hole_kws)
        self.legend_kws = legend_kws or {}

    def render(self, ax):
        colors = self.datasource.get_data("colors")
        labels = self.datasource.get_data("labels")

        for ix, contour in enumerate(self.datasource.polygons):
            outline_kwargs = self.outline_kws.copy()
            if colors is not None:
                outline_kwargs["edgecolor"] = colors[ix]
            outline, holes = self._contour_polygon_patch(
                contour, outline_kwargs=self.outline_kws, hole_kwargs=self.hole_kws
            )
            ax.add_patch(outline)
            for hole in holes:
                ax.add_patch(hole)

            if labels is not None:
                self._label_patch(ax, outline, labels[ix])
        if (self.palette is not None) & (not self.on_zoom_view):
            self.legend = cat_legend(
                colors=self.palette.values(),
                labels=self.palette.keys(),
                **self.legend_kws,
            )


class FilledPolygonRenderPlan(PolygonMixin):
    def __init__(
        self,
        polygons: PolygonDataSource,
        labels: Sequence = None,
        colors: Sequence = None,
        palette: Dict = None,
        color="#FFE31A",
        linewidth: int = 1,
        alpha=0.3,
        legend_kws=None,
        **kwargs,
    ):
        self.palette = palette
        self.alpha = alpha

        if colors is not None:
            colors = [to_rgba(c, alpha) if is_color_like(c) else c for c in colors]
        super().__init__(polygons, labels=labels, colors=colors)

        self.legend_kws = legend_kws or {}
        self.kwargs = dict(
            facecolor=to_rgba(color, alpha), edgecolor=color, linewidth=linewidth
        )
        if kwargs is not None:
            self.kwargs.update(kwargs)

    def render(self, ax):
        colors = self.datasource.get_data("colors")
        labels = self.datasource.get_data("labels")

        for ix, contour in enumerate(self.datasource.polygons):
            kwargs = self.kwargs.copy()
            if colors is not None:
                c = colors[ix]
                kwargs["facecolor"] = c
                kwargs["edgecolor"] = c
            patch = self._filled_polygon_patch(contour, **kwargs)
            ax.add_patch(patch)

            if labels is not None:
                self._label_patch(ax, patch, labels[ix])

        if (self.palette is not None) & (not self.on_zoom_view):
            self.legend = cat_legend(
                colors=self.palette.values(),
                labels=self.palette.keys(),
                **self.legend_kws,
            )


class ZoomMixin:
    def render(self, ax, plans):
        pass


class ZoomRenderPlan(ZoomMixin, RenderPlan):
    # See example:
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html

    def __init__(
        self,
        image_datasource: ImageDataSource,  # Which image to subscribe to
        x_range: tuple[int, int],
        y_range: tuple[int, int],
        anchor: tuple[float, float] = (1.2, 0),
        size: tuple[float, float] = (1, 1),
        axis="on",
        xaxis="top",
        edgecolor="k",
        alpha=0.5,
    ):
        self.image_datasource = image_datasource
        self.x_range = x_range
        self.y_range = y_range
        self.anchor = anchor
        self.size = size
        self.bounds = (anchor[0], anchor[1], size[0], size[1])
        self.axis = axis
        self.xaxis = xaxis
        self.edgecolor = edgecolor
        self.alpha = alpha

    def render(self, ax, plans):
        xmin, xmax = self.x_range
        ymin, ymax = self.y_range

        if all([0 <= x <= 1 for x in [xmin, xmax, ymin, ymax]]):
            w, h = self.image_datasource.get_image_size()
            x_start, y_start = (
                self.image_datasource.viewport.x,
                self.image_datasource.viewport.y,
            )
            xmin, xmax = int(xmin * w) + x_start, int(xmax * w) + x_start
            ymin, ymax = int(ymin * h) + y_start, int(ymax * h) + y_start

        # create a zoomed inset of the image
        axins = ax.inset_axes(self.bounds, xlim=(xmin, xmax), ylim=(ymax, ymin))

        for plan in plans:
            if plan.zoom_view_visible:
                plan.on_zoom_view = True
                plan.render(axins)
                plan.on_zoom_view = False

        _axes_style(axins, anchor=self.anchor, axis=self.axis, xaxis=self.xaxis)

        rect = (xmin, ymin, xmax - xmin, ymax - ymin)
        ax.indicate_inset(
            rect, axins, edgecolor=self.edgecolor, zorder=10, alpha=self.alpha
        )


class WSIViewer:
    """The viewer for whole slide images.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole slide image data.
    in_bounds : bool, default: True
        Whether to render the image in the bounds.
    img_bytes_limit : int, default: 2e9
        The limit of the bytes to render image.
        The image size has a strong impact on the rendering time.

    """

    _viewport: Viewport

    def __init__(
        self,
        wsi: WSIData,
        in_bounds: bool = True,
        img_bytes_limit: int = 2e9,
    ):
        self.wsi = wsi
        self.in_bounds = in_bounds
        self.bytes_limit = img_bytes_limit
        self._render_plans = []
        self._temp_render_plans = []
        self._image_render_plan = None
        self._zoom_image_render_plan = None
        self._zoom_plan = None
        self._is_zoom_cached = True

        # There is only one image source
        self.image_source: ImageDataSource = ImageDataSource(self.wsi.reader)
        # There is only one zoom image source
        self.zoom_image_source: ImageDataSource | None = None
        # There can be multiple tile sources
        self.tile_source: Dict[str, TileDataSource] = {}
        # There can be multiple polygon sources
        self.polygon_source: Dict[str, PolygonDataSource] = {}
        self._has_image = False
        self.title = None

        if self.in_bounds:
            self.set_viewport(*self.wsi.properties.bounds)
        else:
            self.set_viewport(0, 0, *wsi.properties.shape[::-1])

    def set_viewport(self, x, y, w, h):
        """Set the window to (x, y, w, h) at level 0"""
        self._viewport = self._get_viewport(x, y, w, h)

        self.image_source.set_viewport(self._viewport)
        for name, source in self.tile_source.items():
            source.set_viewport(self._viewport)
        for name, source in self.polygon_source.items():
            source.set_viewport(self._viewport)

    def set_tissue_id(self, tissue_id, tissue_key="tissues"):
        """ """
        tissues = self.wsi[tissue_key]
        tissue_geo = tissues[tissues["tissue_id"] == tissue_id].geometry.iloc[0]
        xmin, ymin, xmax, ymax = tissue_geo.bounds
        self.set_viewport(xmin, ymin, xmax - xmin, ymax - ymin)

    def get_render_plans(self, in_zoom=False):
        plans = []
        if not in_zoom:
            if self._image_render_plan is not None:
                plans.append(self._image_render_plan)
        else:
            if self._zoom_image_render_plan is not None:
                plans.append(self._zoom_image_render_plan)
        plans.extend(self._render_plans)
        plans.extend(self._temp_render_plans)
        return plans

    def reset_render_plans(self):
        self._temp_render_plans = []
        if self._is_zoom_cached:
            self._zoom_image_render_plan = None

    def add_image(self, in_zoom=True):
        plan = SlideImageRenderPlan(self.image_source)
        plan.zoom_view_visible = in_zoom
        self._image_render_plan = plan
        self._has_image = True
        return self

    def add_scalebar(
        self,
        label=None,
        length_fraction=None,
        width_fraction=None,
        location="lower right",
        pad=None,
        border_pad=None,
        sep=None,
        frameon=True,
        color="k",
        box_alpha=None,
        box_color="w",
        scale_loc="bottom",
        label_loc="top",
        font_properties=None,
        fixed_value=None,
        fixed_units=None,
        rotation=None,
        in_zoom=True,
        cache=True,
    ):
        """
        Add a scale bar to the plot.

        The scale bar is a matplotlib_scalebar.ScaleBar object.

        Parameters
        ----------
        label : str, optional
            The label of the scale bar.
        length_fraction : float, optional
            The length of the scale bar as a fraction of the axes.
        width_fraction : float, optional
            The width of the scale bar as a fraction of the axes.
        location : {'upper right', 'upper left', 'lower left', 'lower right', 'right', \
            'center left', 'center right', 'lower center', \
            'upper center', 'center', 'best'}, optional
            The location of the scale bar. Same as legend location in matplotlib.
        pad : float, optional
            The padding inside the scale bar.
        border_pad : float, optional
            The padding outside the scale bar.
        sep : float, optional
            The separation between the scale bar and the label.
        frameon : bool, default: True
            Whether to draw a box behind the scale bar.
        color : str, optional
            The color of the scale bar.
        box_alpha : float, optional
            The alpha of the box.
        box_color : str, default: 'w'
            The color of the box.
        scale_loc : {'top', 'bottom', 'left', 'right'}, optional
            Location of the scale with respect to the scale bar.
        label_loc : {'top', 'bottom', 'left', 'right'}, optional
            The location of the label.
        font_properties : dict, optional
            The font properties of the label.
        fixed_value : float, optional
            To fix the value of the scale bar.
        fixed_units : str, optional
            Units of the `fixed_value`.
        rotation : {'horizontal', 'vertical'}, float, optional
            The rotation of the scale bar.

        """

        dx = self.wsi.properties.mpp

        options = dict(
            label=label,
            length_fraction=length_fraction,
            width_fraction=width_fraction,
            location=location,
            pad=pad,
            border_pad=border_pad,
            sep=sep,
            frameon=frameon,
            color=color,
            box_alpha=box_alpha,
            box_color=box_color,
            scale_loc=scale_loc,
            label_loc=label_loc,
            font_properties=font_properties,
            fixed_value=fixed_value,
            fixed_units=fixed_units,
            rotation=rotation,
            # bbox_to_anchor=bbox_to_anchor,
            # bbox_transform=bbox_transform,
        )

        plan = ScaleBarRenderPlan(self.image_source, dx, **options)
        plan.zoom_view_visible = in_zoom
        if cache:
            self._render_plans.append(plan)
        else:
            self._temp_render_plans.append(plan)
        return self

    def mark_origin(self, in_zoom=False, cache=True):
        plan = OriginXYArrowRenderPlan()
        plan.zoom_view_visible = in_zoom
        if cache:
            self._render_plans.append(plan)
        else:
            self._temp_render_plans.append(plan)
        return self

    def _process_polygons(self, key, label_by, color_by, palette):
        shapes = self.wsi[key]
        labels = shapes[label_by] if label_by is not None else None

        colors = None
        if color_by is not None:
            colors = shapes[color_by]
            if isinstance(colors, pd.Categorical):
                # If colors is categorical, convert to codes
                u_colors = colors.codes
            else:
                u_colors = np.unique(colors)
            palette = get_dict_palette(palette, u_colors)
            colors = [palette[c] for c in colors]
        else:
            # Set palette to None
            palette = None

        if self.polygon_source.get(key) is None:
            polygons = shapes.geometry
            ds = PolygonDataSource(polygons)
            ds.set_viewport(self._viewport)
            self.polygon_source[key] = ds

        return self.polygon_source[key], labels, colors, palette

    def add_contours(
        self,
        key,
        label_by=None,
        color_by=None,
        palette: PaletteType = None,
        outline_color: ColorType = "#117554",
        hole_color: ColorType = "#4379F2",
        linewidth: int = 1,
        outline_kwargs: Dict = None,
        hole_kwargs: Dict = None,
        legend_kws: Dict = None,
        legend: bool = True,
        in_zoom: bool = True,
        cache=True,
    ):
        """Add contours to the plot.

        Contours are the boundaries of the polygons.
        If you intend to plot filled polygons, use `add_polygons` instead.

        Parameters
        ----------
        key : str
            The key of the polygon table.
        label_by : str, optional
            The column contains labels for the contours.
            The labels will be shown on top of the contours.
        color_by : str, optional
            The column contains colors or colors encoding (both categorical and continuous) for the contours.
            The color only affects the outline of the contours but not the holes.
        palette : dict, list, str, optional
            The color palette for the colors.
            You can use a dict where the keys are the unique values in the color column and the values are the colors.
            If a list, the colors will be assigned to the unique values in the color column.
            If a color, all the contours will have the same color.
        outline_color : color, default: "#117554"
            The default color of the outline of the contours.
        hole_color : color, default: "#4379F2"
            The default color of the holes of the contours.
        linewidth : int, default: 1
            The width of the contours.
        outline_kwargs : dict, optional
            The keyword arguments for the outline of the contours.
        hole_kwargs : dict, optional
            The keyword arguments for the holes of the contours.
        legend_kws : dict, optional
            The keyword arguments for the legend.
        in_zoom : bool, default: True
            Whether the contours are rendered in the zoom view.
        legend : bool, default: True
            Whether to show the legend.

        """
        contours, labels, colors, palette = self._process_polygons(
            key, label_by, color_by, palette
        )

        plan = ContourRenderPlan(
            contours,
            labels=labels,
            colors=colors,
            palette=palette,
            outline_color=outline_color,
            hole_color=hole_color,
            linewidth=linewidth,
            outline_kws=outline_kwargs,
            hole_kws=hole_kwargs,
            legend_kws=legend_kws,
        )
        plan.zoom_view_visible = in_zoom
        plan.legend_visible = legend
        if cache:
            self._render_plans.append(plan)
        else:
            self._temp_render_plans.append(plan)
        return self

    def add_polygons(
        self,
        key: str,
        label_by: str = None,
        color_by: str = None,
        palette: PaletteType = None,
        alpha: float = 0.3,
        color: ColorType = "#FFE31A",
        linewidth: int = 1,
        legend_kws: Dict = None,
        legend: bool = True,
        in_zoom: bool = True,
        cache=True,
        **kwargs,
    ):
        """Add filled polygons to the plot.

        If you intend to plot only the outlines of the polygons, use `add_contours` instead.

        Parameters
        ----------
        key : str
            The key of the polygon table.
        label_by : str, optional
            The column contains labels for the polygons.
            The labels will be shown on top of the polygons.
        color_by : str, optional
            The column contains colors or colors encoding (both categorical and continuous) for the polygons.
            The color will fill the polygons.
        palette : dict, list, str, optional
            The color palette for the colors.
            You can use a dict where the keys are the unique values in the color column and the values are the colors.
            If a list, the colors will be assigned to the unique values in the color column.
            If a color, all the polygons will have the same color.
        alpha : float, default: 0.3
            The transparency of the polygons.
        color : color, default: "#FFE31A"
            The default color of the polygons.
        linewidth : int, default: 1
            The width of the outline of the polygons.
        legend_kws : dict, optional
            The keyword arguments for the legend.
        in_zoom : bool, default: True
            Whether the polygons are rendered in the zoom view.
        legend : bool, default: True
            Whether to show the legend.

        """
        polygons, labels, colors, palette = self._process_polygons(
            key, label_by, color_by, palette
        )
        plan = FilledPolygonRenderPlan(
            polygons,
            labels=labels,
            colors=colors,
            palette=palette,
            alpha=alpha,
            color=color,
            linewidth=linewidth,
            legend_kws=legend_kws,
            **kwargs,
        )
        plan.zoom_view_visible = in_zoom
        plan.legend_visible = legend
        if cache:
            self._render_plans.append(plan)
        else:
            self._temp_render_plans.append(plan)

        return self

    def _process_tiles(
        self, key, color_by=None, feature_key=None, cmap="coolwarm", palette=None
    ):
        tiles = self.wsi[key]
        tiles_xy = tiles.bounds[["minx", "miny"]].to_numpy()
        spec = self.wsi.tile_spec(key)

        if self.tile_source.get(key) is None:
            ds = TileDataSource(tiles_xy, spec)
            ds.set_viewport(self._viewport)
            self.tile_source[key] = ds

        # If visualize feature
        if color_by is not None:
            if feature_key is not None:
                feature_key = self.wsi._check_feature_key(feature_key, key)
                adata = self.wsi[feature_key]
                if color_by is not None:
                    if color_by in adata.obs.columns:
                        values = adata.obs[color_by].values
                        title = color_by
                    elif color_by in adata.var.index:
                        values = adata[:, color_by].X.flatten()
                        title = f"{feature_key} ({color_by})"
                    else:
                        raise ValueError(
                            f"color_by={color_by} not found in feature of '{feature_key}'."
                        )
                else:
                    raise ValueError(
                        "color_by must be provided when feature_key is provided."
                    )
            # If visualize tile table
            else:
                if color_by not in tiles.columns:
                    raise ValueError(
                        f"color_by={color_by} not found in the tile table."
                    )
                values = tiles[color_by] if color_by is not None else None
                title = color_by
        else:
            values = None
            title = None

        # Decide the color palette of tiles
        is_categorical = False
        if values is not None:
            if isinstance(values, pd.CategoricalDtype):
                is_categorical = True
            elif not isinstance(values[0], Number):
                is_categorical = True

        if is_categorical:
            cats = pd.unique(values)  # Set sorted=False to avoid NA in the data
            palette = get_dict_palette(palette, cats)
        container = dict(
            ds=self.tile_source[key],
            values=values,
            title=title,
            cmap=cmap,
            palette=palette,
            is_categorical=is_categorical,
        )

        return container

    def add_tiles(
        self,
        key,
        feature_key=None,
        color_by=None,
        style: Literal["scatter", "heatmap"] = "scatter",
        palette=None,
        cmap="coolwarm",
        norm=None,
        vmin=None,
        vmax=None,
        alpha=0.5,
        smooth=False,
        smooth_scale=2,
        rasterized=False,
        size=None,
        zoom_size=None,
        marker="o",
        gridcolor: ColorType = "k",
        linewidth=0.1,
        legend_kws=None,
        legend=True,
        in_zoom=True,
        cache=True,
    ):
        """Add tiles to the plot.

        Parameters
        ----------
        key : str
            The key of the tile table.
        feature_key : str, optional
            The key of the feature table to visualize.
        color_by : str, optional
            The column or feature to color the tiles by.
        style : {'scatter', 'heatmap'}, default: 'scatter'
            The style to render the tiles.
        palette : dict, list, str, optional
            The color palette for categorical coloring.
        cmap : str, optional
            The colormap for continuous coloring.
        norm : matplotlib.colors.Normalize, optional
            The normalization for the colormap.
        vmin, vmax : float, optional
            The min and max values for the colormap.
        alpha : float, default: 0.5
            The transparency of the tiles.
        smooth : bool, default: False
            Whether to apply smoothing to the heatmap.
        smooth_scale : int, default: 2
            The scale of the smoothing kernel.
        rasterized : bool, default: False
            Whether to rasterize the scatter plot.
        size : float, optional
            The size of the scatter points.
        zoom_size : float, optional
            The size of the scatter points in zoom view.
        marker : str, default: 'o'
            The marker style for scatter.
        gridcolor : color, default: 'k'
            The color of the grid lines.
        linewidth : float, default: 0.1
            The width of the grid lines.
        legend_kws : dict, optional
            The keyword arguments for the legend.
        legend : bool, default: True
            Whether to show the legend.
        in_zoom : bool, default: True
            Whether to render in the zoom view.
        cache : bool, default: True
            Whether to cache the render plan.

        """
        container = self._process_tiles(key, color_by, feature_key, cmap, palette)
        if color_by is None:
            plan = GridTilesRenderPlan(
                container["ds"], color=gridcolor, linewidth=linewidth
            )
        else:
            if style == "scatter":
                plan = ScatterTilesRenderPlan(
                    container["ds"],
                    container["values"],
                    palette=container["palette"],
                    cmap=container["cmap"],
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha,
                    size=size,
                    zoom_size=zoom_size,
                    rasterized=rasterized,
                    marker=marker,
                    legend_kws=legend_kws,
                )
            elif style == "heatmap":
                plan = HeatmapTilesRenderPlan(
                    container["ds"],
                    self.image_source,
                    container["values"],
                    palette=container["palette"],
                    cmap=cmap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha,
                    smooth=smooth,
                    smooth_scale=smooth_scale,
                    legend_kws=legend_kws,
                )
            else:
                raise ValueError(
                    f"Unknown style={style}, options are 'scatter' or 'heatmap'."
                )
        plan.zoom_view_visible = in_zoom
        plan.legend_visible = legend
        self.title = container["title"]
        if cache:
            self._render_plans.append(plan)
        else:
            self._temp_render_plans.append(plan)

    def add_zoom(
        self,
        xmin: Number | None = None,
        xmax: Number | None = None,
        ymin: Number | None = None,
        ymax: Number | None = None,
        tissue_id: int | None = None,
        tissue_key: str = "tissues",
        anchor: tuple[float, float] = (1.2, 0),
        size: tuple[float, float] = (1, 1),
        edgecolor: ColorType = "k",
        alpha=0.5,
        axis="on",
        xaxis="top",
        cache: bool = True,
    ):
        """Add a zoom window to the plot.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : int, float, optional
            The coordinates of the zoom window.
            You may also provide in (0-1) range to specify the fraction of the window.
        tissue_id : int, optional
            The tissue id to zoom into.
        tissue_key : str, optional
            The key of the tissue table.
        anchor : tuple, default: (1.2, 0)
            The anchor point of the zoom window, relative to the main axes.
        size : tuple, default: (1, 1)
            The size of the zoom window, relative to the main axes.

        """
        if not any([xmin, xmax, ymin, ymax]):
            if tissue_id is None:
                raise ValueError(
                    "Either (xmin, xmax, ymin, ymax) or tissue_id must be provided."
                )
            tissues = self.wsi[tissue_key]
            tissue_geo = tissues[tissues["tissue_id"] == tissue_id].geometry.iloc[0]
            xmin, ymin, xmax, ymax = tissue_geo.bounds
        else:
            if tissue_id is not None:
                warnings.warn(
                    "Both (xmin, xmax, ymin, ymax) and tissue_id are also provided. "
                    "tissue_id will be ignored.",
                    stacklevel=find_stack_level(),
                )

        if all([0 <= x <= 1 for x in [xmin, xmax, ymin, ymax]]):
            current_viewport = self._viewport
            downsample = current_viewport.downsample
            xmin = (current_viewport.x + xmin * current_viewport.w) * downsample
            xmax = (current_viewport.x + xmax * current_viewport.w) * downsample
            ymin = (current_viewport.y + ymin * current_viewport.h) * downsample
            ymax = (current_viewport.y + ymax * current_viewport.h) * downsample
        elif all([x > 1 for x in [xmin, xmax, ymin, ymax]]):
            pass
        else:
            raise ValueError(
                "xmin, xmax, ymin, ymax must be either all in (0-1) range or all > 1."
            )

        viewport = self._get_viewport(xmin, ymin, xmax - xmin, ymax - ymin)
        self.zoom_image_source = ImageDataSource(self.wsi.reader)
        self.zoom_image_source.set_viewport(viewport)

        self._zoom_plan = ZoomRenderPlan(
            self.image_source,
            (xmin, xmax),
            (ymin, ymax),
            anchor=anchor,
            size=size,
            axis=axis,
            xaxis=xaxis,
            edgecolor=edgecolor,
            alpha=alpha,
        )
        self._zoom_image_render_plan = SlideImageRenderPlan(self.zoom_image_source)

        if not cache:
            self._is_zoom_cached = False

    def show(self, ax=None, axis="on", xaxis="top"):
        if ax is None:
            ax = plt.gca()

        legends = []
        for plan in self.get_render_plans():
            plan.render(ax)
            if plan.legend_visible:
                legend = plan.get_legend()
                if legend is not None:
                    legends.append(legend)

        # Set the viewport for axes when no image present
        if not self._has_image:
            w = self._viewport.w * self._viewport.downsample
            h = self._viewport.h * self._viewport.downsample
            xmin = self._viewport.x
            xmax = xmin + w
            ymin = self._viewport.y
            ymax = ymin + h
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
        ax = _axes_style(ax, axis=axis, xaxis=xaxis)

        legend_placement = dict(
            loc="center left", bbox_transform=ax.transAxes, bbox_to_anchor=(1.01, 0.5)
        )
        if self._zoom_plan is not None:
            self._zoom_plan.render(ax, self.get_render_plans(in_zoom=True))
            if not self._is_zoom_cached:
                # If not cached, remove the zoom plan
                self._zoom_plan = None
            legend_placement.update(loc="center right", bbox_to_anchor=(-0.1, 0.5))

        if len(legends) > 0:
            stack_legends = vstack(legends, **legend_placement)
            ax.add_artist(stack_legends)

        if self.title is not None:
            ax.set_title(self.title)

        # remove temp render plans
        self._temp_render_plans = []

        return ax

    def _get_viewport(self, x, y, w, h):
        """Get the (x, y, w, h, level, downsample) for the current window"""
        n_bytes = w * h * 8 * 3  # 8 bytes per pixel, 3 channels
        target_level = 0
        downsample = 1

        dw = w
        dh = h
        if n_bytes > self.bytes_limit:
            dh, dw = self.wsi.properties.shape
            while n_bytes > self.bytes_limit:
                target_level += 1
                if target_level >= self.wsi.properties.n_level:
                    # We've reached the highest level but still exceed the limit
                    # Set to the highest level and adjust dimensions if needed
                    target_level = self.wsi.properties.n_level - 1
                    downsample = self.wsi.properties.level_downsample[target_level]
                    dw = w // downsample
                    dh = h // downsample
                    break

                downsample = self.wsi.properties.level_downsample[target_level]
                dw = w // downsample
                dh = h // downsample
                n_bytes = dw * dh * 8 * 3

        return Viewport(int(x), int(y), int(dw), int(dh), target_level, downsample)


def _axes_style(ax, anchor=None, axis="off", xaxis="top"):
    ax.set_aspect("equal", anchor=anchor)
    ax.axis(axis)
    if xaxis == "top":
        ax.spines["top"].set_position(("outward", 0))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks(np.asarray(ax.get_xlim(), dtype=int))
    ax.yaxis.set_ticks(np.asarray(ax.get_ylim(), dtype=int))
    return ax


MPL_QUAL_PALS = {
    "tab10": 10,
    "tab20": 20,
    "tab20b": 20,
    "tab20c": 20,
    "Set1": 9,
    "Set2": 8,
    "Set3": 12,
    "Accent": 8,
    "Paired": 12,
    "Pastel1": 9,
    "Pastel2": 8,
    "Dark2": 8,
}


def get_dict_palette(palette: PaletteType, category: list) -> Dict:
    """Convert a palette to a dictionary if it is not already.

    The category must be a sequence of unique values.
    """
    if palette is None:
        palette = LAZYSLIDE_PALETTE

    if isinstance(palette, dict):
        return palette
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        if palette in MPL_QUAL_PALS:
            # If the palette is a qualitative palette, call by n_colors
            sel = np.arange(len(category))
        else:
            sel = np.linspace(0, 1, len(category))
        colors = cmap(sel)
        return dict(zip(category, colors))
    elif isinstance(palette, Sequence):
        return dict(zip(category, palette))
    elif is_color_like(palette):
        return {cat: to_rgba(palette) for cat in category}
    else:
        raise ValueError(f"Unsupported palette type: {type(palette)}")
