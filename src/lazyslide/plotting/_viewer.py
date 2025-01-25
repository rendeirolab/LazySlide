from __future__ import annotations

import warnings
from itertools import cycle
from numbers import Number
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from wsidata import WSIData

ADOBE_SPECTRUM = [
    "#0FB5AE",
    "#F68511",
    "#147AF3",
    "#E8C600",
    "#DE3D82",
    "#72E06A",
    "#7E84FA",
    "#7326D3",
    "#008F5D",
    "#CB5D00",
    "#4046CA",
    "#BCE931",
]


def _get_thumbnail(image_arr, max_size=1000):
    x_size, y_size = image_arr.shape[0:2]
    ratio = 1
    if x_size > max_size or y_size > max_size:
        if x_size > y_size:
            ratio = x_size / max_size
            y_shape = int(y_size / ratio)
            thumbnail = cv2.resize(image_arr, dsize=(y_shape, max_size))
        else:
            ratio = y_size / max_size
            x_shape = int(x_size / ratio)
            thumbnail = cv2.resize(image_arr, dsize=(max_size, x_shape))
    else:
        thumbnail = image_arr
    return ratio, thumbnail


def _get_best_level(wsi: WSIData, w, h, render_size=1000):
    n_bytes = w * h * 8 * 3  # 8 bytes per pixel, 3 channels
    mem_level = 0
    # Limit the size to 4GB
    while n_bytes > 4e9:
        mem_level += 1
        downsample = wsi.properties.level_downsample[mem_level]
        w //= downsample
        h //= downsample
        n_bytes = w * h * 8 * 3
        if mem_level >= wsi.properties.n_level:
            break

    render_level = 0
    for render_level, (h, w) in enumerate(wsi.properties.level_shape):
        dh = h - render_size
        dw = w - render_size
        if render_level == 0:
            if dh < 0 and dw < 0:
                raise ValueError("The render size is too big for the image.")
        if dh < 0 and dw < 0:
            break
    # Get the level with the closest size to the render size
    if render_level != 0:
        render_level -= 1

    return max(mem_level, render_level)


def _get_length_string(length_in_um):
    """Return a string with the length in um / mm / cm."""
    if length_in_um <= 500:
        length = length_in_um
        unit = "µm"
    elif length_in_um < 10000:
        length = length_in_um / 1000
        unit = "mm"
    else:
        length = length_in_um / 1000 / 10
        unit = "cm"
    return f"{int(length)} {unit}"


class SlideViewer:
    def __init__(
        self,
        wsi: WSIData,
        tissue_key=None,
        tile_key=None,
        render_size: int | None = None,
        tissue_id: int = None,
    ):
        if tissue_id is not None:
            if tissue_key not in wsi.shapes:
                tissue_id = None

        if tissue_id is None:
            if render_size is None:
                render_size = 1000
            level = _get_best_level(wsi, *wsi.properties.shape, render_size)
            slide_img = wsi.reader.get_level(level)
            level_downsample = wsi.properties.level_downsample[level]
            bounds = None
        else:
            gdf = wsi.shapes[tissue_key]
            gdf = gdf[gdf["tissue_id"] == tissue_id]
            minx, miny, maxx, maxy = gdf.bounds.iloc[0]

            x = int(minx)
            y = int(miny)
            w = int(maxx - minx)
            h = int(maxy - miny)

            # Now the best level is calculated based on the region size
            if render_size is None:
                render_size = max(w * 0.3, h * 0.3, 10000)
            level = _get_best_level(wsi, w, h, render_size)
            level_downsample = wsi.properties.level_downsample[level]
            slide_img = wsi.read_region(
                x, y, w / level_downsample, h / level_downsample, level=level
            )
            bounds = np.asarray([x, y, w, h])

        slide_img[np.where(slide_img == 0)] = 255
        thumbnail_downsample, thumbnail = _get_thumbnail(slide_img, render_size)
        self.level_downsample = level_downsample
        self.thumbnail_downsample = thumbnail_downsample
        self.downsample = thumbnail_downsample * level_downsample
        self.thumbnail = thumbnail
        self.wsi = wsi
        self.render_size = render_size
        self.tissue_key = tissue_key
        self.tissue_id = tissue_id
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = None

        if tile_key in wsi.shapes:
            self.tile_spec = wsi.tile_spec(tile_key)
            self.tile_table = wsi.shapes[tile_key]
            if self.tissue_id is not None:
                self.tile_table = self.tile_table[
                    self.tile_table["tissue_id"] == self.tissue_id
                ]
            self.tile_coords = self.tile_table[["x", "y"]].values.astype(float)
            if self.bounds is not None:
                self.tile_coords -= np.asarray(self.bounds[0:2])
            self.tile_center_coords = self.tile_coords + np.array(
                [self.tile_spec.raw_width / 2, self.tile_spec.raw_height / 2]
            )
            self.tile_key = tile_key
        else:
            self.tile_coords = None
            self.tile_spec = None
            self.tile_key = None
        self._title = None

    def set_title(self, title):
        if title is not None:
            self._title = title

    def add_tissue(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.thumbnail)  # , extent=self.extent)
        ax.set_axis_off()

    def add_scale_bar(self, ax=None, fontsize=7):
        from matplotlib.lines import Line2D

        # add a scale bar
        mpp = self.wsi.properties.mpp
        if mpp is not None:
            possible_scale = np.array(
                [
                    0.1,
                    0.2,
                    0.5,
                    1,
                    2,
                    5,  # noqa: E201
                    10,
                    20,
                    50,
                    100,
                    200,
                    500,  # noqa: E201
                    1000,
                    2000,
                    5000,
                ]
            )  # noqa: E201
            if self.bounds is not None:
                y_size, x_size = self.bounds[2:4]
            else:
                y_size, x_size = self.wsi.properties.shape
            allow_max = x_size / 4  # 25% of the image width
            # the length of scale bar is only relative
            # to the actual image size
            scale_bar_px = possible_scale / mpp
            mask = scale_bar_px < allow_max
            actual_px = scale_bar_px[mask][-1]
            scale = possible_scale[mask][-1]
            if scale >= 1:
                scale = int(scale)
            scale_bar_width = 2
            # since the image is scaled
            # the render size of the scale bar is also scaled
            bar_length = actual_px / x_size

            x_start = 0.05
            bar_y = 0.01
            text_y = 0.02
            line = Line2D(
                [x_start, x_start + bar_length],
                [bar_y, bar_y],
                lw=scale_bar_width,
                color="black",
                transform=ax.transAxes,
            )
            ax.add_line(line)
            ax.text(
                x_start + bar_length / 2,
                text_y,
                _get_length_string(scale),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                transform=ax.transAxes,
            )

    def add_origin(self, ax=None):
        if ax is None:
            ax = plt.gca()
        # Add arrow that is always 10% of the image width / height
        y_size, x_size = self.thumbnail.shape[0:2]
        arrow_length = min(x_size * 0.1, y_size * 0.1)
        arrow_options = dict(
            head_width=arrow_length * 0.2,
            head_length=arrow_length * 0.2,
            fc="k",
            ec="k",
            clip_on=False,
            length_includes_head=True,
        )
        ax.arrow(0, 0, arrow_length, 0, **arrow_options)
        ax.text(arrow_length, 0, "x", ha="left", va="center")

        ax.arrow(0, 0, 0, arrow_length, **arrow_options)
        ax.text(0, arrow_length, "y", ha="center", va="top")

    def add_tiles(
        self,
        alpha=None,
        rasterized=True,
        ax=None,
    ):
        if self.tile_key is None:
            warnings.warn("No tiles found.")
            return
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        if ax is None:
            ax = self.ax

        w, h = (
            self.tile_spec.raw_width / self.downsample,
            self.tile_spec.raw_height / self.downsample,
        )

        rects = []
        for x, y in self.tile_coords / self.downsample:
            rects.append(
                Rectangle((x, y), w, h, fill=False, ec="black", lw=0.5, alpha=alpha)
            )
        ax.add_collection(
            PatchCollection(rects, match_original=True, rasterized=rasterized)
        )

    def add_points(
        self,
        feature_key=None,
        color=None,
        vmin=None,
        vmax=None,
        cmap=None,
        norm=None,
        palette=None,
        alpha=None,
        ax=None,
        marker="o",
        size=None,
        rasterized=True,
        **kwargs,
    ):
        # If there are no tiles, return
        if self.tile_key is None:
            print("No tiles found.")
            return

        from legendkit import cat_legend, colorart

        if ax is None:
            ax = self.ax

        c = None
        add_title = ""
        if feature_key is not None:
            adata = self.wsi.fetch.features_anndata(
                feature_key, self.tile_key, tile_graph=False
            )
            if color is not None:
                if isinstance(color, str):
                    if color in adata.obs.columns:
                        c = adata.obs.loc[:, color].values
                        add_title = color
                    elif color in adata.var.index:
                        c = adata[:, color].X.flatten()
                        add_title = f"{feature_key} feature {color}"
                    else:
                        raise KeyError(
                            f"The requested color `{color}` not found in {feature_key}."
                        )
                elif isinstance(color, int):
                    c = adata[:, color].X.flatten()
                    add_title = f"{feature_key} feature {adata.var.index[color]}"
                else:
                    raise ValueError(
                        f"The requested color `{color}` is invalid in {feature_key}."
                    )
        else:
            if color is not None:
                if color not in self.tile_table:
                    raise KeyError(
                        f"The requested color `{color}` "
                        f"not found in points {self.tile_key}. "
                        f"If you want to visualize features, please provide a feature_key."
                    )
                c = self.tile_table.loc[:, color].values
                add_title = color

        add_cat_legend = False
        add_colorart = False

        # Edge case: no tile
        # Need to check the length of c
        if (c is not None) and (len(c) > 0):
            if not isinstance(c[0], Number):
                # If value is categorical
                entries = np.unique(c)
                if palette is None:
                    palette = cycle(ADOBE_SPECTRUM)
                if isinstance(palette, Iterable):
                    palette = dict(zip(entries, palette))
                c = [palette.get(v, "black") for v in c]
                add_cat_legend = True
            else:
                if cmap is None:
                    cmap = "rainbow"
                add_colorart = True
        else:
            c = "black"

        # The size of the points is based on
        # - The number of points
        # - The size of the plot
        if size is None:
            fig = ax.get_figure()
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            n_points = len(self.tile_coords)
            DENSITY = 50 if self.tissue_id is None else 1000  # heuristic value
            size = bbox.width * bbox.height * DENSITY / n_points

        # The points are drawn at the center of the tiles
        xy = self.tile_center_coords / self.downsample
        kwargs = {"zorder": 10, **kwargs}
        sm = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=c,
            s=size,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            marker=marker,
            edgecolor="none",
            linewidth=0,
            **kwargs,
        )
        sm.set_rasterized(rasterized)
        if add_colorart:
            colorart(sm, width=1, height=8, ax=ax)
        if add_cat_legend:
            cat_legend(
                colors=palette.values(),
                labels=palette.keys(),
                loc="out right center",
                handle=marker,
                ax=ax,
            )
        self.set_title(add_title)

    def add_shapes(
        self,
        key,
        ax,
        color,
        hole_color,
        linewidth,
        alpha,
        show_bbox=False,
        show_shape=True,
    ):
        from matplotlib.patches import Rectangle

        if key in self.wsi.shapes:
            shapes = self.wsi.shapes[key]
            if self.tissue_id is not None:
                if "tissue_id" in shapes.columns:
                    shapes = shapes[shapes["tissue_id"] == self.tissue_id]
            for c in shapes.geometry:
                if show_bbox:
                    # Draw bbox of shape
                    bbox = np.array(c.bounds)
                    if self.bounds is not None:
                        bbox -= np.array(self.bounds[0:2])
                    minx, miny, maxx, maxy = bbox / self.downsample
                    ax.add_patch(
                        Rectangle(
                            (minx, miny),
                            maxx - minx,
                            maxy - miny,
                            fill=False,
                            ec=color,
                            lw=linewidth,
                        )
                    )
                if show_shape:
                    holes = [np.asarray(h.coords) for h in c.interiors]
                    c = np.array(c.exterior.coords, dtype=np.float32)
                    if self.bounds is not None:
                        c -= np.array(self.bounds[0:2])
                        for h in holes:
                            h -= np.array(self.bounds[0:2])
                    c /= self.downsample
                    for h in holes:
                        h /= self.downsample
                    ax.plot(
                        c[:, 0],
                        c[:, 1],
                        color=color,
                        alpha=alpha,
                        lw=linewidth,
                    )

                    for h in holes:
                        ax.plot(
                            h[:, 0],
                            h[:, 1],
                            color=hole_color,
                            alpha=alpha,
                            lw=linewidth,
                        )

        return ax

    def add_contours_holes(
        self,
        contour_color="#117554",
        hole_color="#4379F2",
        show_bbox=False,
        show_shape=True,
        linewidth=2,
        alpha=None,
        ax=None,
    ):
        if ax is None:
            ax = plt.gca()

        tissue_key = self.tissue_key
        self.add_shapes(
            tissue_key,
            ax,
            contour_color,
            hole_color,
            linewidth,
            alpha,
            show_bbox=show_bbox,
            show_shape=show_shape,
        )

    def _draw_shapes_anno(self, key, ax, fmt=None, **kwargs):
        default_style = dict(
            fontsize=10,
            color="black",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", pad=2, lw=1),
        )
        kwargs = {**default_style, **kwargs}
        if key in self.wsi.shapes:
            shapes = self.wsi.shapes[key]
            if self.tissue_id is not None:
                shapes = shapes[shapes["tissue_id"] == self.tissue_id]
            for _, row in shapes.iterrows():
                minx, miny, maxx, maxy = row.geometry.bounds
                x = minx + (maxx - minx) / 2
                y = miny
                if self.bounds is not None:
                    x = x - self.bounds[0]
                    y = y - self.bounds[1]
                x, y = x / self.downsample, y / self.downsample
                tissue_id = row.tissue_id
                text = tissue_id
                if fmt is not None:
                    text = fmt.format(tissue_id)
                _ = ax.text(x, y, text, **kwargs)

                # calculate the bbox
                # renderer = ax.figure.canvas.get_renderer()
                # bbox = t.get_bbox_patch().get_tightbbox(renderer)
                # display_y = ax.transData.transform((0, y))[1]
                # print(display_y, bbox.y0)
                # offset_px = bbox.y0 - display_y + renderer.points_to_pixels(3)
                # offset_y = ax.transAxes.inverted().transform((0, offset_px))
                # print(offset_y, t.get_position())
                # t.set_position((x, y + offset_y[1]))
                # print(t.get_position())

    def add_tissue_id(self, ax=None, fmt=None, **kwargs):
        if ax is None:
            ax = self.ax

        tissue_key = self.tissue_key
        self._draw_shapes_anno(tissue_key, ax, fmt=fmt, **kwargs)

    def add_title(self, title, ax=None):
        if ax is None:
            ax = self.ax
        if title is not None:
            ax.set_title(title)
        elif self._title is not None:
            ax.set_title(self._title)

    def save(self, filename, **kwargs):
        kwargs.update(dict(bbox_inches="tight"))
        self.fig.savefig(filename, **kwargs)
