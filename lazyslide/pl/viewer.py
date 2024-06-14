from itertools import cycle
from numbers import Number
from typing import Sequence

import cv2
import numpy as np
import matplotlib.pyplot as plt

from lazyslide import WSI

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
        if tissue_id is None:
            # TODO: When there is only ONE LEVEL
            if level == "auto":
                level = _get_best_level(wsi, render_size)
            level_downsample = wsi.metadata.level_downsample[level]
            slide_img = wsi.reader.get_level(level)
            slide_img[np.where(slide_img == 0)] = 255
            bounds = None
        else:
            gdf = wsi.sdata.shapes[f"{tissue_key}_contours"]
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

        # TODO: Only get tiles when tile_key is not None
        self.tile_key = tile_key
        if self.tile_key is None:
            self.tile_coords = (
                wsi.sdata.points[tile_key][["x", "y"]].compute().to_numpy()
            )
            self.tile_spec = wsi.get_tile_spec(tile_key)

    def add_tissue(self, title=None, ax=None, scale_bar=True):
        from matplotlib.lines import Line2D

        if ax is None:
            ax = plt.gca()
        ax.imshow(self.thumbnail)  # , extent=self.extent)
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title)
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
        value=None,
        cmap=None,
        norm=None,
        palette=None,
        alpha=None,
        ax=None,
        title=None,
    ):
        from legendkit import colorart
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        if ax is None:
            ax = self.ax
        if title is not None:
            ax.set_title(title)

        w, h = (
            self.tile_spec.ops_width / self.downsample,
            self.tile_spec.ops_height / self.downsample,
        )

        rects = []
        # If value is not None, color the tiles
        if value is not None:
            default_cmap = "bwr"
            # If value is continuous
            if isinstance(value[0], Number):
                cmap = default_cmap if cmap is None else cmap
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(value)
                tiles_colors = sm.to_rgba(value)
                colorart(sm, ax=ax)
            else:
                # If value is categorical
                entries = np.unique(value)
                if palette is None:
                    palette = cycle(ADOBE_SPECTRUM)
                if isinstance(palette, Sequence):
                    palette = dict(zip(entries, palette))
                tiles_colors = [palette.get(v, "black") for v in value]

            for (x, y), c in zip(self.tile_coords / self.downsample, tiles_colors):
                rects.append(Rectangle((x, y), w, h, fc=c, ec="none", alpha=alpha))
        # Else, just draw the tiles
        else:
            for x, y in self.tile_coords / self.downsample:
                rects.append(
                    Rectangle((x, y), w, h, fill=False, ec="black", lw=0.5, alpha=alpha)
                )
        ax.add_collection(PatchCollection(rects, match_original=True))

    def _draw_cnt(self, key, ax, color, linewidth, alpha):
        if key in self.wsi.sdata.shapes:
            shapes = self.wsi.sdata.shapes[key]
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
            shapes = self.wsi.sdata.shapes[key]
            if self.tissue_id is not None:
                shapes = shapes[shapes["tissue_id"] == self.tissue_id]
            for _, row in shapes.iterrows():
                centroid = row.geometry.centroid
                x, y = centroid.x / self.downsample, centroid.y / self.downsample
                tissue_id = row.tissue_id
                text = tissue_id
                if fmt is not None:
                    text = fmt.format(tissue_id)
                ax.text(x, y, text, **kwargs)

    def add_tissue_id(self, ax=None, fmt=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        tissue_key = self.tissue_key
        self._draw_cnt_anno(f"{tissue_key}_contours", ax, **kwargs)

    def save(self, filename, **kwargs):
        kwargs.update(dict(bbox_inches="tight"))
        self.fig.savefig(filename, **kwargs)
