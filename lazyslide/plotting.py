from itertools import cycle
from numbers import Number
from typing import Mapping, Iterable

import cv2
import numpy as np
from legendkit import cat_legend, colorart
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from lazyslide.utils import create_mesh_array

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
            ratio = max_size / x_size
            y_shape = int(y_size * ratio)
            thumbnail = cv2.resize(image_arr, dsize=(y_shape, max_size))
        else:
            ratio = max_size / y_size
            x_shape = int(x_size * ratio)
            thumbnail = cv2.resize(image_arr, dsize=(max_size, x_shape))
    else:
        thumbnail = image_arr
    return ratio, thumbnail


class SlideViewer:
    def __init__(self, slide_img, max_size=1000, ax=None, fig=None, scale_factor=1):
        downsample_ratio, thumbnail = _get_thumbnail(slide_img, max_size)

        if ax is None and fig is None:
            fig, ax = plt.subplots()
        elif ax is None:
            ax = plt.gca()
        elif fig is None:
            fig = ax.get_figure()
        self.fig = fig
        self.ax = ax
        self.downsample_ratio = downsample_ratio * scale_factor
        self.thumbnail = thumbnail
        self.max_size = max_size
        self.legend = None
        self.tissue_shape = thumbnail.shape[0:2]
        self.extent = [0, self.tissue_shape[1], self.tissue_shape[0], 0]

    def add_tissue(self, title=None, ax=None):
        if ax is None:
            ax = self.ax
        ax.imshow(self.thumbnail)  # , extent=self.extent)
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title)

    def add_mask(self, mask, cmap="gray", alpha=0.5, ax=None):
        if ax is None:
            ax = self.ax
        _, mask_thumbnail = _get_thumbnail(mask, max_size=self.max_size)
        ax.imshow(mask_thumbnail, cmap=cmap, alpha=alpha, extent=self.extent)
        ax.set_axis_off()

    def add_origin(self, ax=None):
        if ax is None:
            ax = self.ax
        ox, upper_x = ax.get_xlim()
        oy, upper_y = ax.get_ylim()
        arrow_length = upper_x * 0.05

        arrow_options = dict(
            head_width=arrow_length * 0.2,
            head_length=arrow_length * 0.2,
            fc="k",
            ec="k",
            clip_on=False,
            length_includes_head=True,
        )
        # x arrow
        ax.arrow(ox, upper_y, arrow_length, 0, **arrow_options)
        ax.text(arrow_length, upper_y, "x", ha="left", va="center")
        # y arrow
        ax.arrow(ox, upper_y, 0, arrow_length, **arrow_options)
        ax.text(ox, arrow_length, "y", ha="center", va="top")

    def add_tiles(
        self,
        tiles,
        tile_ops,
        value=None,
        cmap=None,
        norm=None,
        palette=None,
        alpha=None,
        ax=None,
        title=None,
    ):
        if ax is None:
            ax = self.ax
        if title is not None:
            ax.set_title(title)
        coords = tiles * self.downsample_ratio

        tile_h = tile_ops.ops_height * self.downsample_ratio
        tile_w = tile_ops.ops_width * self.downsample_ratio

        img_shape = self.thumbnail.shape[0:2]
        if value is None:
            # v_arr = create_mesh_array(img_shape, coords, tile_h, tile_w)
            # v_arr = np.ma.masked_invalid(v_arr)
            # ax.pcolorfast(v_arr, norm=norm, cmap=cmap, alpha=alpha)
            rects = []
            for x, y in coords:
                rects.append(
                    Rectangle(
                        (x, y),
                        tile_w,
                        tile_h,
                        fill=False,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                )
            ax.add_collection(PatchCollection(rects, match_original=True, alpha=alpha))
        else:
            # check value is numeric or categorical
            isnumeric = isinstance(value[0], Number)
            if isnumeric:
                v_arr = create_mesh_array(img_shape, coords, tile_h, tile_w, value)
                v_arr = np.ma.masked_invalid(v_arr)
                cm = ax.pcolorfast(v_arr, norm=norm, cmap=cmap, alpha=alpha)
                colorart(cm, ax=ax)
            else:
                kinds = np.unique(value)
                kinds_encode = {k: i for i, k in enumerate(kinds)}
                kinds_colors = {k: c for k, c in zip(kinds, cycle(ADOBE_SPECTRUM))}
                if palette is not None:
                    if isinstance(palette, str):
                        cmap = plt.get_cmap(palette, len(kinds))
                        colors = cmap(np.arange(len(kinds)))
                        kinds_colors = {k: c for k, c in zip(kinds, colors)}
                    elif isinstance(palette, Mapping):
                        kinds_colors = palette
                        kinds = list(kinds_colors.keys())
                    elif isinstance(palette, Iterable):
                        kinds_colors = {k: c for k, c in zip(kinds, cycle(palette))}
                    else:
                        raise ValueError("palette must be str, Mapping or Iterable")
                encode_arr = np.array([kinds_encode[v] for v in value])
                v_arr = create_mesh_array(img_shape, coords, tile_h, tile_w, encode_arr)
                v_arr = np.ma.masked_invalid(v_arr)
                cmap = ListedColormap([kinds_colors[k] for k in kinds])
                ax.pcolorfast(v_arr, norm=norm, cmap=cmap, alpha=alpha)
                cat_legend(
                    labels=kinds,
                    colors=[kinds_colors[k] for k in kinds],
                    ax=ax,
                    loc="out right center",
                    title=title,
                )

    def add_contours_holes(
        self,
        contours,
        holes,
        contour_color="green",
        hole_color="blue",
        linewidth=1,
        alpha=None,
        ax=None,
    ):
        if ax is None:
            ax = self.ax
        ratio = self.downsample_ratio
        for contour in contours:
            ax.plot(
                contour[:, 0] * ratio,
                contour[:, 1] * ratio,
                color=contour_color,
                alpha=alpha,
                lw=linewidth,
            )
        for hole in holes:
            ax.plot(
                hole[:, 0] * ratio,
                hole[:, 1] * ratio,
                color=hole_color,
                alpha=alpha,
                lw=linewidth,
            )

    def save(self, filename, **kwargs):
        kwargs.update(dict(bbox_inches="tight"))
        self.fig.savefig(filename, **kwargs)
