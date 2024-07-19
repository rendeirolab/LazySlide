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
    """
    Display the tissue image.

    Parameters
    ----------
    wsi : WSI
        The whole-slide image object.
    level : int or str, default: "auto"
        The level to extract the tissue image from.
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
    render_size : int, default: 1000
        The size of the rendered image.
    ax : matplotlib.axes.Axes, default: None
        The axes to plot on.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.WSI("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissue(wsi)
        >>> zs.pl.tissue(wsi)

    """
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
    level : int or str, default: "auto"
        The level to extract the tissue image from.
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
        Additional keyword arguments for plotting.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.WSI("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissue(wsi)
        >>> zs.pp.tiles(wsi, 256, mpp=0.5)
        >>> zs.pl.tiles(wsi, tissue_id=0, show_grid=True, color='contrast')

    """
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
        slide.add_tiles(ax=ax)
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
