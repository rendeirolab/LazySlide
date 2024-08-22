from wsi_data import WSIData
from lazyslide._const import Key
from .api import tiles, tissue


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

    n_tissue = wsi.n_tissue(tissue_key)
    # Slide overview
    h, w = wsi.properties.shape
    slide_ax = pw.Brick(figsize=(h / w, 1))
    tissue(wsi, ax=slide_ax, show_origin=False)
    slide_ax.set_title(wsi.reader.file)

    # Plot tissue
    n_tis = wsi.n_tissue(tissue_key)
    tissue_table = wsi.sdata.shapes[tissue_key]

    tissue_ax = None
    for metric in tissue_metrics:
        ax = pw.Brick(figsize=(1, 1))
        sns.barplot(data=tissue_table, x="tissue_id", y=metric, ax=ax, width=0.5)
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
            ax = pw.Brick(figsize=(1, 1))
            tiles(
                wsi,
                tile_key=tile_key,
                tissue_id=tid,
                color=metric,
                ax=ax,
                show_origin=False,
            )
            t_ax |= ax

        if tile_ax is None:
            tile_ax = t_ax
        else:
            tile_ax /= t_ax

    tile_stat_ax = None
    tile_table = wsi.sdata.shapes[tile_key][tile_metrics + ["tissue_id"]]
    for metric in tile_metrics:
        ax = pw.Brick(figsize=(2, n_tissue))
        sns.stripplot(data=tile_table, x="tissue_id", y=metric, ax=ax, size=1)
        ax.set(xlabel="Tissue ID", ylabel="", title=metric)
        if tile_stat_ax is None:
            tile_stat_ax = ax
        else:
            tile_stat_ax |= ax

    return (slide_ax / tissue_ax) | (tile_ax / tile_stat_ax)
