import warnings
from itertools import cycle
from numbers import Number
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lazyslide import WSI
from lazyslide.utils import check_feature_key

ADOBE_SPECTRUM = [
    "#0FB5AE",
    "#F68511",
    "#4046CA",
    "#7326D3",
    "#147AF3",
    "#72E06A",
    "#7E84FA",
    "#DE3D82",
    "#008F5D",
    "#CB5D00",
    "#E8C600",
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


def _get_best_level(wsi: WSI, render_size=1000):
    h_arr, w_arr = [], []
    level_bytes = []
    for h, w in wsi.metadata.level_shape:
        h_arr.append(h)
        w_arr.append(w)
        level_bytes.append(h * w)
    # Get the level with the closest size to the render size
    h_arr = np.abs(np.array(h_arr) - render_size)
    w_arr = np.abs(np.array(w_arr) - render_size)
    level = min(np.argmin(h_arr), np.argmin(w_arr))
    # If the level is too big to fit into memory, get the previous level
    while level_bytes[level] > 1e9:
        if level_bytes[level + 1] > 1e9:
            level -= 1
        else:
            level += 1
    return level


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
        wsi: WSI,
        level="auto",
        render_size=1000,
        tissue_key="tissue",
        tissue_id=None,
        tile_key="tiles",
    ):
        if tissue_id is not None:
            if f"{tissue_key}_contours" not in wsi.sdata.shapes:
                tissue_id = None

        if tissue_id is None:
            # TODO: When there is only ONE LEVEL
            if level == "auto":
                level = _get_best_level(wsi, render_size)
            level_downsample = wsi.metadata.level_downsample[level]
            slide_img = wsi.reader.get_level(level)
            slide_img[np.where(slide_img == 0)] = 255
            bounds = None
        else:
            gdf = wsi.get_shape_table(f"{tissue_key}_contours")
            gdf = gdf[gdf["tissue_id"] == tissue_id]
            minx, miny, maxx, maxy = gdf.bounds.iloc[0]

            x = int(minx)
            y = int(miny)
            w = int(maxx - minx)
            h = int(maxy - miny)

            if level == "auto":
                # Now the best level is calculated based on the region size
                y_size, x_size = wsi.metadata.shape
                region_ratio = x_size / w
                level = _get_best_level(wsi, render_size * region_ratio)
            level_downsample = wsi.metadata.level_downsample[level]

            slide_img = wsi.reader.get_region(
                x, y, w / level_downsample, h / level_downsample, level=level
            )
            slide_img[np.where(slide_img == 0)] = 255
            bounds = np.asarray([x, y, w, h])

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

        if tile_key in wsi.sdata.points:
            self.tile_spec = wsi.get_tile_spec(tile_key)
            self.tile_table = wsi.get_tiles_table(tile_key)
            if self.tissue_id is not None:
                self.tile_table = self.tile_table[
                    self.tile_table["tissue_id"] == self.tissue_id
                ]
            self.tile_coords = self.tile_table[["x", "y"]].values
            if self.bounds is not None:
                self.tile_coords -= np.array(self.bounds[0:2])
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

    def add_tissue(self, ax=None, scale_bar=True):
        from matplotlib.lines import Line2D

        if ax is None:
            ax = plt.gca()
        ax.imshow(self.thumbnail)  # , extent=self.extent)
        ax.set_axis_off()
        # add a scale bar
        if scale_bar:
            mpp = self.wsi.metadata.mpp
            if mpp is not None:
                possible_scale = np.array(
                    [
                        0.1,
                        0.2,
                        0.5,
                        1,
                        2,
                        5,
                        10,
                        20,
                        50,
                        100,
                        200,
                        500,
                        1000,
                        2000,
                        5000,
                    ]
                )  # um/px
                if self.bounds is not None:
                    y_size, x_size = self.bounds[2:4]
                else:
                    y_size, x_size = self.wsi.metadata.shape
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
                    fontsize=7,
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
        ax.add_collection(PatchCollection(rects, match_original=True))

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
        size=50,
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
            table_key = check_feature_key(self.wsi, feature_key, self.tile_key)
            adata = self.wsi.get_features(feature_key, self.tile_key)[
                self.tile_table.index
            ]
            if adata.shape[0] != self.tile_table.shape[0]:
                raise ValueError(
                    f"Seems like your tiles in {self.tile_key} has changed, "
                    f"please rerun zs.tl.feature_extraction."
                )
                return
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
                            f"The requested color `{color}` not found in tables {table_key}."
                        )
                elif isinstance(color, int):
                    c = adata[:, color].X.flatten()
                    add_title = f"{feature_key} feature {adata.var.index[color]}"
                else:
                    raise ValueError(f"The requested color `{color}` is invalid.")
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
            **kwargs,
        )
        sm.set_rasterized(rasterized)
        if add_colorart:
            colorart(sm, ax=ax)
        if add_cat_legend:
            cat_legend(
                colors=palette.values(),
                labels=palette.keys(),
                loc="out right center",
                handle=marker,
                ax=ax,
            )
        self.set_title(add_title)

    def _draw_cnt(self, key, ax, color, linewidth, alpha):
        if key in self.wsi.sdata.shapes:
            shapes = self.wsi.get_shape_table(key)
            if self.tissue_id is not None:
                shapes = shapes[shapes["tissue_id"] == self.tissue_id]
            for c in shapes.geometry:
                c = np.array(c.exterior.coords, dtype=np.float32)
                if self.bounds is not None:
                    c -= np.array(self.bounds[0:2])
                c /= self.downsample
                ax.plot(
                    c[:, 0],
                    c[:, 1],
                    color=color,
                    alpha=alpha,
                    lw=linewidth,
                )
        return ax

    def add_contours_holes(
        self,
        contour_color="green",
        hole_color="blue",
        linewidth=1,
        alpha=None,
        ax=None,
    ):
        if ax is None:
            ax = plt.gca()

        tissue_key = self.tissue_key
        self._draw_cnt(f"{tissue_key}_contours", ax, contour_color, linewidth, alpha)
        self._draw_cnt(f"{tissue_key}_holes", ax, hole_color, linewidth, alpha)

    def _draw_cnt_anno(self, key, ax, fmt=None, **kwargs):
        default_style = dict(
            fontsize=10,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white"),
        )
        kwargs = {**default_style, **kwargs}
        if key in self.wsi.sdata.shapes:
            shapes = self.wsi.get_shape_table(key)
            if self.tissue_id is not None:
                shapes = shapes[shapes["tissue_id"] == self.tissue_id]
            for _, row in shapes.iterrows():
                centroid = row.geometry.centroid
                x = centroid.x
                y = centroid.y
                if self.bounds is not None:
                    x = x - self.bounds[0]
                    y = y - self.bounds[1]
                x, y = x / self.downsample, y / self.downsample
                tissue_id = row.tissue_id
                text = tissue_id
                if fmt is not None:
                    text = fmt.format(tissue_id)
                ax.text(x, y, text, **kwargs)

    def add_tissue_id(self, ax=None, fmt=None, **kwargs):
        if ax is None:
            ax = self.ax

        tissue_key = self.tissue_key
        self._draw_cnt_anno(f"{tissue_key}_contours", ax, fmt=fmt, **kwargs)

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
