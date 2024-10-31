from itertools import cycle

from wsidata import WSIData
from lazyslide._const import Key
from ._api import tiles, tissue

from ._viewer import ADOBE_SPECTRUM


def qc_summary(
    wsi: WSIData,
    tissue_metrics: list[str],
    tile_metrics: list[str],
    tissue_key: str = Key.tissue,
    tile_key: str = Key.tiles,
):
    try:
        import patchworklib as pw
        import seaborn as sns
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Please install patchworklib with `pip install patchworklib`."
        )

    n_tissue = wsi.fetch.n_tissue(tissue_key)
    # Slide overview
    h, w = wsi.properties.shape
    slide_ax = pw.Brick(figsize=(h / w, 1))
    tissue(wsi, ax=slide_ax, show_origin=False)
    slide_ax.set_title(wsi.reader.file)

    # Plot tissue
    n_tis = wsi.fetch.n_tissue(tissue_key)
    tissue_table = wsi.shapes[tissue_key]

    tissue_ax = None
    for metric, c in zip(tissue_metrics, cycle(ADOBE_SPECTRUM)):
        ax = pw.Brick(figsize=(1, 1))
        if metric in {"brightness", "redness"}:
            ax.set_ylim(0, 255)
        sns.barplot(
            data=tissue_table, x="tissue_id", y=metric, ax=ax, width=0.5, color=c
        )
        ax.set(xlabel="Tissue ID", ylabel="", title=metric)
        if tissue_ax is None:
            tissue_ax = ax
        else:
            tissue_ax |= ax

    tile_ax = None
    for tid in range(n_tis):
        t_ax = pw.Brick(figsize=(1, 1))
        tissue(
            wsi,
            tissue_id=tid,
            ax=t_ax,
            show_origin=False,
            show_id=False,
            title=f"Tissue {tid}",
        )

        for metric in tile_metrics:
            vmin, vmax = None, None
            if metric == "focus":
                vmin, vmax = 0, 14
            elif metric == "contrast":
                vmin, vmax = 0, 1
            ax = pw.Brick(figsize=(1, 1))
            tiles(
                wsi,
                tile_key=tile_key,
                tissue_id=tid,
                color=metric,
                ax=ax,
                show_origin=False,
                vmin=vmin,
                vmax=vmax,
            )
            t_ax |= ax

        if tile_ax is None:
            tile_ax = t_ax
        else:
            tile_ax /= t_ax

    tile_stat_ax = None
    tile_table = wsi.shapes[tile_key][tile_metrics + ["tissue_id"]]
    for metric, c in zip(tile_metrics, cycle(ADOBE_SPECTRUM)):
        ax = pw.Brick(figsize=(2, n_tissue))
        if metric == "focus":
            ax.axhline(4, color="r", linestyle="--")
            ax.set_ylim(0, 14)
        elif metric == "contrast":
            ax.axhline(0.5, color="r", linestyle="--")
            ax.set_ylim(0, 1)
        sns.stripplot(data=tile_table, x="tissue_id", y=metric, ax=ax, size=2, color=c)
        ax.set(xlabel="Tissue ID", ylabel="", title=metric)
        if tile_stat_ax is None:
            tile_stat_ax = ax
        else:
            tile_stat_ax |= ax

    return (slide_ax | tile_ax) / (tissue_ax | tile_stat_ax)
