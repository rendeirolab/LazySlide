from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wsidata import WSIData

from .._const import Key
from ._wsi_viewer import WSIViewer


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
    zoom=None,
    img_bytes_limit=2e9,
    ax=None,
    ncols=4,
    wspace=0.5,
    hspace=0.5,
    return_figure=False,
):
    """
    Display the tissue image.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    tissue_id : int or 'all', default: None
        The tissue id (piece) to extract.
    tissue_key : str, default: "tissue"
        The tissue key.
    title : str or array of str, default: None
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
    zoom : (xmin, xmax, ymin, ymax), default: None
        A zoom view for the current viewport.
        If in range [0, 1], will be interpreted as a fraction of the image size.
        If > 1, will be interpreted as the absolute size in pixels.
    img_bytes_limit : int, default: 2e9
        The image bytes limits.
    ax : matplotlib.axes.Axes, default: None
        The axes to plot on.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.gtex_artery()
        >>> zs.pl.tissue(wsi, tissue_id="all")

    """
    # We need to prepare the following variables
    # axes, list of ax
    # tissue_id, list of tissue ids

    # Prepare tissue_id
    if tissue_key in wsi:
        if tissue_id is None:
            tissue_ids = [None]
        elif isinstance(tissue_id, int):
            tissue_ids = [tissue_id]
        elif tissue_id == "all":
            tissue_ids = sorted(wsi[tissue_key].tissue_id.unique().tolist())
        else:
            tissue_ids = list(tissue_id)
    else:
        tissue_ids = [None]

    # Prepare title
    if title is None:
        if tissue_ids[0] is None:
            titles = [""]
        else:
            titles = [f"Tissue {tid}" for tid in tissue_ids]
    elif isinstance(title, str):
        titles = [title] * len(tissue_ids)
    else:
        titles = list(title)

    # Prepare axes for plotting
    n_axes = len(tissue_ids)
    if n_axes == 1:
        if ax is None:
            ax = plt.gca()
        axes = [ax]
    else:
        nrows = n_axes // int(ncols) + 1
        figure = plt.figure(figsize=(ncols * 4, nrows * 4))
        gs = GridSpec(nrows, ncols, wspace=wspace, hspace=hspace)
        axes = [figure.add_subplot(gs[i]) for i in range(n_axes)]

    for tid, t, ax in zip(tissue_ids, titles, axes):
        viewer = WSIViewer(
            wsi,
            in_bounds=in_bounds,
            img_bytes_limit=img_bytes_limit,
        )
        viewer.add_image()
        if show_contours and tissue_key in wsi:
            viewer.add_contours(
                key=tissue_key,
                label_by="tissue_id" if show_id else None,
            )
        if tid is not None:
            viewer.set_tissue_id(tid, tissue_key=tissue_key)
        if scalebar:
            viewer.add_scalebar()
        if mark_origin:
            viewer.mark_origin()
        if zoom is not None:
            viewer.add_zoom(*zoom)
        viewer.title = t
        viewer.show(ax=ax)

    # Return the axes if there is only one
    if n_axes > 1 and return_figure:
        return figure
    return None


def tiles(
    wsi: WSIData,
    feature_key=None,
    color=None,
    tissue_id=None,
    tissue_key=None,
    tile_key=Key.tiles,
    title=None,
    style: Literal["scatter", "heatmap"] = "heatmap",
    show_image=True,
    show_contours=True,
    show_id=False,
    mark_origin=True,
    scalebar=True,
    in_bounds=True,
    img_bytes_limit=2e9,
    zoom=None,
    alpha=0.9,
    smooth=False,
    smooth_scale=2,
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
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    feature_key : str, default: None
        The feature key assigned when generating the numeric tile features.
    color : str, default: None
        The column/ feature name that should be visualized from feature_key.
        If feature_key is None, this is the column name from the tiles table.
    tissue_id : int or 'all', default: None
        The tissue id (piece) to plot.
        If None, all will be plotted.
    tissue_key : str, default: "tissue"
        The tissue key.
    tile_key : str, default: "tiles"
        The key of the tiles in the :bdg-danger:`shapes` slot.
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
    scalebar : bool, default: True
        Show the scalebar.
    zoom : (xmin, xmax, ymin, ymax), default: None
        A zoom view for the current viewport.
        If in range [0, 1], will be interpreted as a fraction of the image size.
        If > 1, will be interpreted as the absolute size in pixels.
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
    rasterized : bool, default: False
        Rasterize the points.
    kwargs : dict
        Additional keyword arguments for plotting.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.tl.tile_prediction(wsi, "contrast")
        >>> zs.pl.tiles(wsi, tissue_id=0, color='contrast')

    """
    tile_spec = wsi.tile_spec(tile_key)
    # Prepare tissue_id
    if tissue_key is None:
        tissue_key = tile_spec.tissue_name

    if tissue_key in wsi:
        if tissue_id is None:
            tissue_ids = [None]
        elif isinstance(tissue_id, int):
            tissue_ids = [tissue_id]
        elif tissue_id == "all":
            tissue_ids = sorted(wsi[tissue_key].tissue_id.unique().tolist())
        else:
            tissue_ids = list(tissue_id)
    else:
        tissue_ids = [None]

    # Prepare colors
    if color is None:
        colors = [None]
    elif isinstance(color, str):
        colors = [color]
    else:
        colors = list(color)

    # Prepare axes for plotting
    n_axes = len(tissue_ids) * len(colors)
    if n_axes == 1:
        if ax is None:
            ax = plt.gca()
        axes = [ax]
    else:
        nrows = n_axes // int(ncols) + 1
        figure = plt.figure(figsize=(ncols * 4, nrows * 4))
        gs = GridSpec(nrows, ncols, wspace=wspace, hspace=hspace)
        axes = [figure.add_subplot(gs[i]) for i in range(n_axes)]

    # Prepare title
    if title is None:
        titles = [""] * n_axes
    else:
        titles = list(title)

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
        smooth=smooth,
        smooth_scale=smooth_scale,
        rasterized=rasterized,
        gridcolor=gridcolor,
        linewidth=linewidth,
        **kwargs,
    )

    ix = 0
    for tid in tissue_ids:
        for c in colors:
            t = titles[ix]
            ax = axes[ix]
            ix += 1

            viewer = WSIViewer(
                wsi, in_bounds=in_bounds, img_bytes_limit=img_bytes_limit
            )
            if show_image:
                viewer.add_image()

            if show_contours and tissue_key in wsi:
                viewer.add_contours(
                    key=tissue_key,
                    label_by="tissue_id" if show_id else None,
                )
            if tid is not None:
                viewer.set_tissue_id(tid)
            if mark_origin:
                viewer.mark_origin()
            if scalebar:
                viewer.add_scalebar()
            if zoom is not None:
                viewer.add_zoom(*zoom)

            viewer.add_tiles(
                key=tile_key,
                color_by=c,
                feature_key=feature_key,
                **options,
            )
            if t == "":
                if tid is not None:
                    t += f"Tissue ({tid})"
                if viewer.title is not None:
                    t += f" {viewer.title}"
            viewer.title = t
            viewer.show(ax=ax)


def annotations(
    wsi: WSIData,
    key: str,
    color=None,
    label=None,
    show_image=True,
    mark_origin=True,
    scalebar=True,
    in_bounds=True,
    img_bytes_limit=2e9,
    tissue_key=Key.tissue,
    tissue_id=None,
    zoom=None,
    fill=True,
    linewidth=0.5,
    palette=None,
    alpha=0.5,
    legend_kws=None,
    legend=True,
    title=None,
    ncols=4,
    wspace=0.5,
    hspace=0.5,
    ax=None,
):
    """
    Display the annotations.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    key : str
        The annotation key.
    color : str, optional
        The column or feature name to use for coloring the annotations.
    label : str, optional
        The column or feature name to use for labeling the annotations.
    show_image : bool, default: True
        Whether to display the underlying tissue image.
    mark_origin : bool, default: True
        Whether to mark the origin on the plot.
    scalebar : bool, default: True
        Whether to show a scalebar.
    in_bounds : bool, default: True
        Whether to restrict annotations to the image bounds.
    img_bytes_limit : int, default: 2e9
        The maximum number of bytes for the image.
    tissue_key : str, default: "tissue"
        The key for tissue segmentation.
    tissue_id : int or 'all', optional
        The tissue id(s) to display annotations for.
    zoom : tuple, optional
        (xmin, xmax, ymin, ymax) for zooming into a region.
    fill : bool, default: True
        Whether to fill the annotation polygons.
    linewidth : float, default: 0.5
        The line width of the annotation polygons.
    palette : str, optional
        The color palette to use.
    alpha : float, default: 0.5
        The transparency of the annotation polygons.
    legend_kws : dict, optional
        Additional keyword arguments for the legend.
    legend : bool, default: True
        Whether to display a legend.
    title : str or list of str, optional
        The title(s) for the plot(s).
    ncols : int, default: 4
        Number of columns for subplot arrangement.
    wspace : float, default: 0.5
        Width space between subplots.
    hspace : float, default: 0.5
        Height space between subplots.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.

    Returns
    -------
    None or matplotlib.figure.Figure
        The figure if multiple axes are created and return_figure is True, otherwise None.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pl.annotations(wsi, key="annotations", tissue_id="all")

    """
    # Prepare tissue_id
    if tissue_key in wsi:
        if tissue_id is None:
            tissue_ids = [None]
        elif isinstance(tissue_id, int):
            tissue_ids = [tissue_id]
        elif tissue_id == "all":
            tissue_ids = sorted(wsi[tissue_key].tissue_id.unique().tolist())
        else:
            tissue_ids = list(tissue_id)
    else:
        tissue_ids = [None]

    # Prepare title
    if title is None:
        if tissue_ids[0] is None:
            titles = [""]
        else:
            titles = [f"Tissue {tid}" for tid in tissue_ids]
    elif isinstance(title, str):
        titles = [title] * len(tissue_ids)
    else:
        titles = list(title)

    # Prepare axes for plotting
    n_axes = len(tissue_ids)
    if n_axes == 1:
        if ax is None:
            ax = plt.gca()
        axes = [ax]
    else:
        nrows = n_axes // int(ncols) + 1
        figure = plt.figure(figsize=(ncols * 4, nrows * 4))
        gs = GridSpec(nrows, ncols, wspace=wspace, hspace=hspace)
        axes = [figure.add_subplot(gs[i]) for i in range(n_axes)]
    for tid, t, ax in zip(tissue_ids, titles, axes):
        viewer = WSIViewer(wsi, in_bounds=in_bounds, img_bytes_limit=img_bytes_limit)
        if show_image:
            viewer.add_image()
        if mark_origin:
            viewer.mark_origin()
        if scalebar:
            viewer.add_scalebar()
        if tissue_id is not None:
            viewer.set_tissue_id(tid)
        if fill:
            viewer.add_polygons(
                key,
                color_by=color,
                label_by=label,
                palette=palette,
                alpha=alpha,
                legend=legend,
                legend_kws=legend_kws,
                linewidth=linewidth,
            )
        else:
            viewer.add_contours(
                key,
                color_by=color,
                label_by=label,
                palette=palette,
                legend=legend,
                legend_kws=legend_kws,
                linewidth=linewidth,
            )

        if zoom is not None:
            viewer.add_zoom(*zoom)
        viewer.title = t
        viewer.show(ax=ax)
