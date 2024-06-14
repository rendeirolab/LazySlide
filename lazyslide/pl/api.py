import matplotlib.pyplot as plt

from lazyslide.wsi import WSI
from .viewer import SlideViewer


# TODO: Plot tissue ID at the center of tissue piece
def tissue(
    wsi: WSI,
    level="auto",
    tissue_id=None,
    tissue_key="tissue",
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


def features(
    wsi: WSI,
    level="auto",
    tissue_id=None,
    tissue_key="tissue",
    tile_key="tiles",
    feature_key=None,
    feature_name=None,
    render_size=1000,
    alpha=0.5,
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
        tile_key=tile_key,
    )
    slide.add_tissue(ax=ax)
    value = None
    if feature_key is not None:
        if feature_name is None:
            feature_name = 0
        value = wsi.sdata.tables[f"{tile_key}/{feature_key}"][
            :, feature_name
        ].X.flatten()
    slide.add_tiles(ax=ax, value=value, alpha=alpha)
