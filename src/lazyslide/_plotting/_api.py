from typing import Literal, Iterable

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wsidata import WSIData

from ._wsi_viewer import WSIViewer
from .._const import Key


def thumbnail(wsi: WSIData, ax=None):
    """
    Display the thumbnail.

    Parameters
    ----------
    wsi : WSI
        The whole-slide image object.
    ax : matplotlib.axes.Axes, default: None
        The axes to plot on.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.open_wsi("")


    """
    viewer = WSIViewer(wsi)
    viewer.add_image()
    viewer.show(ax=ax, axis="off")


def tissue(
    wsi: WSIData,
    tissue_id=None,
    tissue_key=Key.tissue,
    title=None,
    show_contours=True,
    show_id=True,
    mark_origin=True,
    scalebar=True,
    in_bounds=True,
    img_bytes_limit=4e9,
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
    mark_origin : bool, default: True
        Show the origin.
    show_id : bool, default: True
        Show the tissue id.
    scalebar : bool, default: True
        Show the scalebar.
    in_bounds : bool, default: True
        Show the tissue in bounds.
    img_bytes_limit : int, default: 4e9
        The image bytes limits.
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
    viewer = WSIViewer(
        wsi,
        in_bounds=in_bounds,
        img_bytes_limit=img_bytes_limit,
    )
    viewer.add_image()

    if tissue_key is not None:
        if show_contours:
            viewer.add_contours(
                key=tissue_key,
                label_by="tissue_id" if show_id else None,
            )
        if tissue_id is not None:
            viewer.set_tissue_id(tissue_id)

    if scalebar:
        viewer.add_scalebar()
    if mark_origin:
        viewer.mark_origin()
    viewer.title = title
    viewer.show(ax=ax)


def tiles(
    wsi: WSIData,
    feature_key=None,
    color=None,
    tissue_id=None,
    tissue_key=Key.tissue,
    tile_key=Key.tiles,
    title=None,
    style: Literal["scatter", "heatmap"] = "heatmap",
    show_image=True,
    show_contours=True,
    show_id=False,
    mark_origin=True,
    scalebar=True,
    alpha=0.9,
    marker="o",
    vmin=None,
    vmax=None,
    cmap=None,
    norm=None,
    palette=None,
    size=None,
    gridcolor="k",
    linewidth=0.1,
    ax=None,
    figure=None,
    rasterized=True,
    ncols=4,
    wspace=0.5,
    hspace=0.5,
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
    style : {"heatmap", "scatter"}, default: "heatmap"
        The style of the plot.
    show_image :  bool, default: True
        Show the tissue image.
    show_contours : bool, default: True
        Show the tissue contours.
    mark_origin : bool, default: True
        Show the origin.
    show_id : bool, default: False
        Show the tissue (piece) id.
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
    figure : matplotlib.figure.Figure, default: None
        The figure to plot on.
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
        >>> zs.pl.tiles(wsi, tissue_id=0, color='contrast')

    """

    viewer = WSIViewer(wsi)
    if show_image:
        viewer.add_image()

    viewer.title = title
    if tissue_key in wsi:
        if show_contours:
            viewer.add_contours(
                key=tissue_key,
                label_by="tissue_id" if show_id else None,
            )
        if tissue_id is not None:
            viewer.set_tissue_id(tissue_id)
    if mark_origin:
        viewer.mark_origin()
    if scalebar:
        viewer.add_scalebar()

    if color is not None:
        if isinstance(color, str):
            color = [color]
        elif isinstance(color, Iterable):
            color = list(color)
        else:
            color = [color]

    options = dict(
        style=style,
        alpha=alpha,
        marker=marker,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        norm=norm,
        palette=palette,
        size=size,
        rasterized=rasterized,
        gridcolor=gridcolor,
        linewidth=linewidth,
        **kwargs,
    )

    if color is not None:
        if len(color) > 1:
            nrows = len(color) // int(ncols) + 1
            if figure is None:
                figure = plt.figure(figsize=(ncols * 4, nrows * 4))
            gs = GridSpec(nrows, ncols, wspace=wspace, hspace=hspace)
            for i, c in enumerate(color):
                ax = figure.add_subplot(gs[i])
                viewer.add_tiles(
                    key=tile_key,
                    feature_key=feature_key,
                    color_by=c,
                    cache=False,
                    **options,
                )
                viewer.show(ax=ax)
        else:
            viewer.add_tiles(
                key=tile_key,
                color_by=color[0],
                feature_key=feature_key,
                **options,
            )
            viewer.show(ax=ax)
    else:
        viewer.add_tiles(
            key=tile_key,
            feature_key=feature_key,
            **options,
        )
        viewer.show(ax=ax)


def annotations(
    wsi: WSIData,
    key: str,
    ax=None,
):
    pass
