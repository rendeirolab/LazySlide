import matplotlib.pyplot as plt

from lazyslide.wsi import WSI
from .viewer import SlideViewer


def tissue(
    wsi: WSI,
    level="auto",
    tissue_id=None,
    tissue_key="tissue",
    title=None,
    show_contours=True,
    show_origin=True,
    show_id=True,
    render_size=1000,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots()
    slide = SlideViewer(
        wsi,
        level=level,
        render_size=render_size,
        tissue_key=tissue_key,
        tissue_id=tissue_id,
    )
    slide.add_tissue(ax=ax)
    if show_origin:
        slide.add_origin(ax=ax)
    if show_id:
        slide.add_tissue_id(ax=ax)
    if show_contours:
        slide.add_contours_holes(ax=ax)
    slide.add_title(title, ax=ax)


def tiles(
    wsi: WSI,
    feature_key=None,
    color=None,
    level="auto",
    tissue_id=None,
    tissue_key="tissue",
    tile_key="tiles",
    title=None,
    show_tissue=True,
    show_point=True,
    show_grid=False,
    show_contours=True,
    show_origin=True,
    show_id=False,
    render_size=1000,
    alpha=0.9,
    marker="o",
    vmin=None,
    vmax=None,
    cmap=None,
    norm=None,
    palette=None,
    size=50,
    ax=None,
    rasterized=False,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()
    slide = SlideViewer(
        wsi,
        level=level,
        render_size=render_size,
        tissue_key=tissue_key,
        tissue_id=tissue_id,
        tile_key=tile_key,
    )
    if show_tissue:
        slide.add_tissue(ax=ax)
    if show_grid:
        slide.add_tiles(rasterized=rasterized, ax=ax)
    if show_origin:
        slide.add_origin(ax=ax)
    if show_id:
        slide.add_tissue_id(ax=ax)
    if show_contours:
        slide.add_contours_holes(ax=ax)
    if show_point:
        slide.add_points(
            feature_key=feature_key,
            color=color,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            norm=norm,
            palette=palette,
            size=size,
            marker=marker,
            ax=ax,
            rasterized=rasterized,
            **kwargs,
        )
    slide.add_title(title, ax=ax)
