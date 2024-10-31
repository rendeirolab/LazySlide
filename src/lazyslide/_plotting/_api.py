import matplotlib.pyplot as plt

from wsidata import WSIData
from ._viewer import SlideViewer
from .._const import Key


def tissue(
    wsi: WSIData,
    tissue_id=None,
    tissue_key=Key.tissue,
    title=None,
    show_contours=True,
    show_origin=True,
    show_id=True,
    show_bbox=False,
    render_size=None,
    scale_bar=True,
    ax=None,
):
    """
    Display the tissue image.

    Parameters
    ----------
    wsi : WSI
        The whole-slide image object.
    tissue_id : int, default: None
        The tissue id (piece) to extract.
    tissue_key : str, default: "tissue"
        The tissue key.
    title : str, default: None
        The title of the plot.
    show_contours : bool, default: True
        Show the tissue contours.
    show_origin : bool, default: True
        Show the origin.
    show_id : bool, default: True
        Show the tissue id.
    render_size : int, default: None
        The size of the rendered image.
        Increase this value for better image quality.
    ax : matplotlib.axes.Axes, default: None
        The axes to plot on.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pl.tissue(wsi)

    """
    if ax is None:
        _, ax = plt.subplots()
    slide = SlideViewer(
        wsi,
        render_size=render_size,
        tissue_key=tissue_key,
        tissue_id=tissue_id,
    )
    slide.add_tissue(ax=ax)
    if scale_bar:
        slide.add_scale_bar(ax=ax)
    if show_origin:
        slide.add_origin(ax=ax)
    if show_id:
        slide.add_tissue_id(ax=ax)
    if show_contours or show_bbox:
        slide.add_contours_holes(ax=ax, show_bbox=show_bbox, show_shape=show_contours)
    slide.add_title(title, ax=ax)


def tiles(
    wsi: WSIData,
    feature_key=None,
    color=None,
    tissue_id=None,
    tissue_key=Key.tissue,
    tile_key=Key.tiles,
    title=None,
    show_tissue=True,
    show_point=True,
    show_grid=False,
    show_contours=True,
    show_origin=True,
    show_id=False,
    show_bbox=False,
    render_size=None,
    alpha=0.9,
    marker="o",
    vmin=None,
    vmax=None,
    cmap=None,
    norm=None,
    palette=None,
    size=None,
    ax=None,
    rasterized=False,
    **kwargs,
):
    """
    Display the tiles.

    Parameters
    ----------
    wsi : WSI
        The whole-slide image object.
    feature_key : str, default: None
        The feature key assigned when generating the numeric tile features.
    color : str, default: None
        The column/ feature name that should be visualized from feature_key.
        If feature_key is None, this is the column name from the tiles table.
    tissue_id : int, default: None
        The tissue id (piece) to plot.
        If None, all will be plotted.
    tissue_key : str, default: "tissue"
        The tissue key.
    tile_key : str, default: "tiles"
        The tile key.
    title : str, default: None
        The title of the plot.
    show_tissue :  bool, default: True
        Show the tissue image.
    show_point : bool, default: True
        Show the points.
        By default, the points are black.
    show_grid : bool, default: False
        Show the tiles grid.
    show_contours : bool, default: True
        Show the tissue contours.
    show_origin : bool, default: True
        Show the origin.
    show_id : bool, default: False
        Show the tissue (piece) id.
    render_size : int, default: 1000
        The size of the rendered image.
    alpha : float, default: 0.9
        The transparency of the points.
    marker : str, default: "o"
        The marker of the points.
    vmin : float, default: None
        The minimum value of the color map.
    vmax : float, default: None
        The maximum value of the color map.
    cmap : str, default: None
        The color map.
    norm : matplotlib.colors.Normalize, default: None
        The normalization of the color map.
    palette : str, default: None
        The color palette.
    size : int, default: 50
        The size of the points.
    ax : matplotlib.axes.Axes, default: None
        The axes to plot on.
    rasterized : bool, default: False
        Rasterize the points.
    kwargs : dict
        Additional keyword arguments for _plotting.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.pp.tiles_qc(wsi, scorers=["contrast"])
        >>> zs.pl.tiles(wsi, tissue_id=0, show_grid=True, color='contrast')

    """
    if ax is None:
        _, ax = plt.subplots()
    slide = SlideViewer(
        wsi,
        render_size=render_size,
        tissue_key=tissue_key,
        tissue_id=tissue_id,
        tile_key=tile_key,
    )
    if show_tissue:
        slide.add_tissue(ax=ax)
    if show_origin:
        slide.add_origin(ax=ax)
    if show_id:
        slide.add_tissue_id(ax=ax)
    if show_contours or show_bbox:
        slide.add_contours_holes(ax=ax, show_bbox=show_bbox, show_shape=show_contours)
    if show_grid:
        slide.add_tiles(rasterized=rasterized, ax=ax)
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
